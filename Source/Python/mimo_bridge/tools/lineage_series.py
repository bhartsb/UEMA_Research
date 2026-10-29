# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Gauges over the running model's whole lineage, read incrementally from its stats files.

The development dashboard's "Over time" tab draws these series and its "Now" tab's
gain-of-function panel reads the same definitions, so a number means one thing on both.

Which files. The lineage is the running run followed back through every resume
(development_dashboard.ancestry_bounds). Each ancestor's records are kept only up to the
step its child resumed from: records a parent wrote after that step belong to a discarded
future (a fork) and are dropped. Runs that are not on the chain are never read.

How it stays cheap. Each stats file is read from the byte where the previous read stopped;
the offset, the file's size and modification time and the extracted rows are cached per
file under Saved/mimo_bridge/lineage_series/. A file whose size and modification time are
unchanged is not opened; a file that grew is read from its cached offset; a file whose first
4,096 bytes changed, or that shrank below the offset, is read again from the start. A line
without its final newline is a write in progress and is left for the next read. Each cache file
carries a code-version tag in its name and its contents (cache_tag: the caller's signature and
this file's source), and a cache with another tag is never read, so a server running older code
and a newer build keep separate caches instead of overwriting each other's.

Missing is unmeasured. A record that does not carry a field gives None for it, never 0:
fields were born on different segments, and every DESIGN.md section 3 block is written only
while its flag is on (docs/gain_of_function_v3/DESIGN.md).

Numbers a person reads (DESIGN.md section 11). Every series carries a DISPLAY form - a count
per 100, a real unit, or a percentage between two named ends - and every sentence this module
writes is built from it (tools/plain_numbers.py), so no page shows a bare score decimal, a
cosine, a correlation or a raw float. The raw values stay in the JSON for the charts, which
transform them with the same form (display a * value + b).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from .plain_numbers import clean, fixed, pct, per

CACHE_VERSION = 5
MAX_BUCKETS = 320
_LOCK = threading.Lock()
REPO = Path(__file__).resolve().parents[4]
MEASUREMENTS = REPO / "docs/gain_of_function_v3/measurements"


def file_tag(path):
    """The first 16 hex digits of the SHA-1 of a source file's bytes, '' when it cannot be read.
    Taken at import, so it names the code this process runs, not a later edit of the file."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


# The code that turns a stats line into cached rows lives in this file: the reader, the default
# extractor and every series definition. Its tag goes into every cache's name and contents
# (cache_tag), so a changed definition never serves rows the old code extracted (verify-fix-dashboards
# 2026-09-23, open issue: the cache names carried no code version and two versions overwrote each
# other's cache). The price: the first read after any edit of this file re-reads the lineage once.
SOURCE_TAG = file_tag(__file__)


# ------------------------------------------------------------------ small readers
def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _g(record, *path):
    cur = record
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and -len(cur) <= key < len(cur):
            cur = cur[key]
        else:
            return None
    return cur


def _field(*path):
    return lambda r: _num(_g(r, *path))


def _first_field(*paths):
    """The first of several field paths that the record carries (a renamed field)."""
    def get(r):
        for path in paths:
            v = _num(_g(r, *path))
            if v is not None:
                return v
        return None
    return get


def _ratio_gain(err_path, base_path):
    """1 - error/baseline: 0 = no better than the baseline, 1 = exact; None when either is absent."""
    def get(r):
        err, base = _num(_g(r, *err_path)), _num(_g(r, *base_path))
        if err is None or base is None or base <= 0:
            return None
        return 1.0 - err / base
    return get


def _rolls(record):
    """|roll| of every body with a valid reading: 0 = on its back, 90 = on its side, 180 = belly down."""
    rows = record.get("body_orientation")
    out = []
    if isinstance(rows, list):
        for body in rows:
            if isinstance(body, dict) and body.get("valid"):
                roll = _num(body.get("roll_degrees"))
                if roll is not None:
                    out.append(min(180.0, abs(roll)))
    return out


def _mean(values):
    return sum(values) / len(values) if values else None


def _list_item(path, index):
    def get(r):
        values = _g(r, *path)
        if isinstance(values, list) and -len(values) <= index < len(values):
            return _num(values[index])
        return None
    return get


def _weight_shares(block):
    """{channel: its weight over the sum of weights} from the wellness_rank block: the producer's
    weight_share_by_channel when present, else computed from `weights`."""
    if not isinstance(block, dict):
        return None
    ws = block.get("weight_share_by_channel")
    if isinstance(ws, dict) and ws:
        return {k: _num(v) for k, v in ws.items() if _num(v) is not None}
    w = block.get("weights")
    if isinstance(w, dict):
        vals = {k: _num(v) for k, v in w.items() if _num(v) is not None}
        total = sum(v for v in vals.values() if v > 0)
        if total > 0:
            return {k: v / total for k, v in vals.items()}
    return None


def lead_channel(record):
    """(channel, its share of the winning goal's lead, its share of the weights) for the channel
    carrying the largest share of the lead, or None."""
    block = _g(record, "wellness_rank")
    shares = _g(block, "winner_margin_share")
    if not isinstance(shares, dict):
        return None
    vals = [(k, _num(v)) for k, v in shares.items() if _num(v) is not None]
    if not vals:
        return None
    name, lead = max(vals, key=lambda kv: kv[1])
    weights = _weight_shares(block) or {}
    return name, lead, weights.get(name)


def _liveness_parts(record):
    """[(block, part, kind 'units'|'pool', share)] from every liveness report in a record: the
    top-level `liveness` block ({layers, pools, line, samples, revived}) and any block's
    `liveness` {layer: share} and `pool_spread` {pool: share} (field contract, 2026-09-23)."""
    out = []
    if not isinstance(record, dict):
        return out
    top = record.get("liveness")
    if isinstance(top, dict):
        for kind, key in (("units", "layers"), ("pool", "pools")):
            for name, v in (top.get(key) or {}).items():
                if _num(v) is not None:
                    out.append(("model", str(name), kind, _num(v)))
    for block, value in record.items():
        if block == "liveness" or not isinstance(value, dict):
            continue
        for kind, key in (("units", "liveness"), ("pool", "pool_spread")):
            sub = value.get(key)
            if isinstance(sub, dict):
                for name, v in sub.items():
                    if _num(v) is not None:
                        out.append((block, str(name), kind, _num(v)))
    return out


LIVE_LINE = 0.5          # DESIGN.md section 10: a layer with under half its units live is flagged
# What "responds" means, in the words every page uses (fix-readout-dashboards, 2026-09-23). DESIGN.md
# section 10 counts a hidden unit as live on a sample when its input rises above -3. A raw input value of
# a hidden unit is not a number a reader can place (the readout scanner's "unit input" rule, commit
# 22b047f9b), so the pages say what the line means, in the words the explainer and the probes use
# (tools/gof_probe_harness.LIVE_RULE, commit 8e200d47e). Computed from the GELU formula x * Phi(x): at
# an input of -3 the slope is -0.012, so the output moves by 1.2% of a change in the input (for a large
# input it moves 100%); further down the low end the slope only shrinks (at -4 it is 0.0005). The
# development dashboard's Now card and the Over-time liveness card print it.
RESPONDS = ("a unit responds when its input leaves the flat low end of its activation function (GELU, the curve "
            "that turns a unit's input into its output), where a change in the input moves the output by less "
            "than 1.2% of the change (100% = the output moves as much as the input)")


def _below_line(record):
    parts = [p for p in _liveness_parts(record) if p[2] == "units"]
    return None if not parts else float(sum(1 for p in parts if p[3] < LIVE_LINE))


def _min_live(record):
    parts = [p[3] for p in _liveness_parts(record) if p[2] == "units"]
    return min(parts) if parts else None


# each sense as the model's registry spells it, the word the explainer uses too (the operator's rule:
# never rename a part); the glossary defines each (development_dashboard_terms.js)
SENSES = (("vision", "vision"), ("tactile", "tactile"), ("proprioception", "proprioception"))
ALL_SENSES = SENSES + (("audio", "audio"),)


# ------------------------------------------------------------------ measured references
def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


# The physics-only goal lines of following are read from this case only: tools/whole_body_following
# on the repaired gauge (env.py and the tool from commit b55ee9a4d on: the eyes' row found by name,
# the chest scored, each order read over exactly its 25 frames). The case "physics_only_as_whole_
# body_following_is_written" and scale_anchors.json whole_body_following (2026-09-19) were measured
# on the gauge that took the last row, the chest, for the eyes, so they are never read as a goal
# line (fix-rows-r2-lineage_series, 2026-09-23).
FOLLOW_CASE = "physics_only_on_the_repaired_gauge"


def gauge_parts(layout):
    """The body parts the live following gauge scores in a body layout, by NAME: every part but the
    eyes (env.py _command_response skips the row named 'gaze', row 23 in both layouts)."""
    from mimo_bridge.body_layout import names_for
    return [name for name in names_for(layout) if name != "gaze"]


def _part_word(name):
    return "the eyes" if name == "gaze" else name


def check_parts(case):
    """(parts, None) when a stored physics reference scored exactly the parts the live gauge scores in
    its body layout; (None, why) otherwise, so a reference over another part set is never a goal line."""
    layout, parts = case.get("layout"), case.get("parts_scored")
    if not layout or not isinstance(parts, list) or not parts:
        return None, "the stored physics reference names no body layout or no list of the parts it scored"
    try:
        want = gauge_parts(layout)
    except ValueError:
        return None, f"the stored physics reference names an unknown body layout ({layout})"
    if sorted(parts) != sorted(want):
        missing = [p for p in want if p not in parts]
        extra = [p for p in parts if p not in want]
        return None, ("the stored physics reference scored other body parts than the live gauge, which scores every "
                      "part but the eyes" + (f"; it left out {', '.join(missing)}" if missing else "")
                      + (f"; it also scored {', '.join(_part_word(p) for p in extra)}" if extra else ""))
    return list(parts), None


def following_reference(path=None):
    """The body-follows-its-order references, from measurements/following_out_of_100.json.

    Its headline (2026-09-23): out of 100 body-part orders, how many end closer to the ordered
    orientation 250 ms later - live on ckpt_step_344400, on physics alone for a reachable posture and
    for independent random turns, and the chance line (the order ignored). The goal lines on the
    trainer's direction-match gauges come from case FOLLOW_CASE only: the physics arms on the 250 ms
    held gauge (gauge_*) and on the one-frame 10 ms gauge that follow_bout averages (frame_*). When that
    case is absent, or scored another set of body parts than the live gauge, every gauge reference is
    None and `gauge_why` says why, so the pages print "no validated success target"; there is no fallback."""
    d = _read_json(Path(path) if path else MEASUREMENTS / "following_out_of_100.json", {}) or {}
    head = d.get("headline") or {}
    out = {"source": None, "gauge_reachable": None, "gauge_independent": None, "frame_reachable": None,
           "frame_independent": None, "gauge_parts": None, "gauge_layout": None, "gauge_angle": None,
           "gauge_source": None, "gauge_why": None}
    if head:
        out.update(source="docs/gain_of_function_v3/measurements/following_out_of_100.json",
                   measured=d.get("measured"),
                   live=_num((head.get("live_model_orders") or {}).get("closer_out_of_100")),
                   reachable=_num((head.get("physics_only_reachable_pose") or {}).get("closer_out_of_100")),
                   independent=_num((head.get("physics_only_independent_random_turns") or {}).get("closer_out_of_100")),
                   chance=_num((head.get("chance_order_ignored") or {}).get("closer_out_of_100")),
                   by_author={k: _num((v or {}).get("closer_out_of_100"))
                              for k, v in (d.get("live_by_author") or {}).items()})
    case = (d.get("cases") or {}).get(FOLLOW_CASE)
    if not isinstance(case, dict):
        out["gauge_why"] = ("no physics-only reference measured on the repaired gauge is stored (measurements/"
                            "following_out_of_100.json has no case " + FOLLOW_CASE + ")")
        return out
    parts, why = check_parts(case)
    if parts is None:
        out["gauge_why"] = why
        return out
    reach, ind = case.get("reachable_pose") or {}, case.get("independent_turns") or {}
    out.update(gauge_reachable=_num(reach.get("held_250ms_mean")), gauge_independent=_num(ind.get("held_250ms_mean")),
               frame_reachable=_num(reach.get("one_frame_10ms_mean")),
               frame_independent=_num(ind.get("one_frame_10ms_mean")),
               gauge_parts=parts, gauge_layout=case.get("layout"), gauge_angle=_num(case.get("angle_deg")),
               gauge_source=("measurements/following_out_of_100.json, case " + FOLLOW_CASE + ", measured "
                             + str(case.get("measured"))))
    return out


def independent_turns_text(ref):
    """'25 independent 20-degree turns' built from the stored reference's own part list and angle."""
    parts, angle = ref.get("gauge_parts"), ref.get("gauge_angle")
    if not parts or angle is None:
        return None
    return f"{len(parts)} independent {fixed(angle, 2)}-degree turns, one for every body part but the eyes"


def follow_lines(ref):
    """The goal line and reference each following series draws, each on its own gauge: the 250 ms
    held gauge (cmd_response_cos_held, and the synergy's following) and the one-frame 10 ms gauge
    (follow_bout). A gauge with no stored physics value gets no line and a reason, never the other
    gauge's value (fix-rows-r2-lineage_series: follow_bout drew the 250 ms goal line)."""
    turns = independent_turns_text(ref)
    parts = ref.get("gauge_parts") or []
    src = ref.get("gauge_source") or ""
    out = dict(goal=None, refs=[], goal_why=ref.get("gauge_why"), bout_goal=None, bout_refs=[], bout_goal_why=None)
    if ref.get("gauge_reachable") is not None:
        out["goal"] = (ref["gauge_reachable"], f"a posture the body really reached, commanded on physics alone, without "
                                               f"any model; the {len(parts)} body parts scored are every one but the "
                                               f"eyes ({src})")
        out["goal_why"] = None
    if ref.get("gauge_independent") is not None and turns:
        out["refs"] = [(ref["gauge_independent"], turns + ", physics alone")]
    if ref.get("frame_reachable") is not None:
        out["bout_goal"] = (ref["frame_reachable"], f"a posture the body really reached, commanded on physics alone, "
                                                    f"read on the same one-frame (10 ms) gauge this series averages; "
                                                    f"the {len(parts)} body parts scored are every one but the eyes "
                                                    f"({src})")
    elif ref.get("gauge_reachable") is not None:
        out["bout_goal_why"] = ("the physics-only goal line was measured on the 250 ms held-command gauge; this series "
                                "averages the one-frame (10 ms) reading, and no physics-only one-frame goal line is "
                                "stored")
    else:
        out["bout_goal_why"] = ref.get("gauge_why")
    if ref.get("frame_independent") is not None and turns:
        out["bout_refs"] = [(ref["frame_independent"], turns + ", physics alone, one-frame gauge")]
    return out


FOLLOW = following_reference()
# Learning progress read on frozen weights, where no learning can happen (b3-wellness measurement
# 2, 2026-09-22, 2 bodies, 320 moments): the channel read -0.016, -0.055 and -0.071 per 100-moment
# window from the fresh world changing alone (proprioception alone -0.20). A reading inside this
# band is not evidence of learning until the same-moment control exists (field contract:
# learning_progress.control_mean).
LP_FROZEN_BAND = (-0.071, -0.016)
# The last lineage step before the gain-of-function build (ckpt_step_344400, DESIGN.md D3). The
# action-use line of the headline is the spread the lineage's own report windows showed up to it:
# single windows are noisy (tactile read -0.084 to +0.046 over H84's 3,462 windows, median 0.00004;
# measured 2026-09-23 from the lineage's stats), so the newest segment's MEDIAN is set against the
# middle 90 of every 100 pre-build windows, never one window against a segment summary.
PRE_BUILD_STEP = 344_400


def action_use_summary(root, bounds, cache_dir=None):
    """{sense: {now, n_now, lo, hi, n_pre}}: the newest segment's median action use over its
    windows, and the 5th to 95th percentile of every window up to PRE_BUILD_STEP. Reads the
    lineage through the same incremental cache as /api/lineage."""
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir else root / "Saved/mimo_bridge/lineage_series"
    sig = signature()
    cols = {s: list(SERIES).index(f"act.{s}") for s, _ in SENSES}
    runs = list(bounds)
    pre, now = {s: [] for s in cols}, {s: [] for s in cols}
    for i, run in enumerate(runs):
        limit = bounds[run]
        rows, _meta, _st = read_rows(Path(run) / "replay_training_stats.jsonl", cache_dir, sig=sig)
        for step, _stamp, values in rows:
            if limit is not None and step > limit:
                continue
            for s, c in cols.items():
                v = values[c] if c < len(values) else None
                if v is None:
                    continue
                if step <= PRE_BUILD_STEP:
                    pre[s].append(v)
                if i == 0:
                    now[s].append(v)
    out = {}
    for s in cols:
        p, n = sorted(pre[s]), sorted(now[s])
        if not n:
            continue
        out[s] = dict(now=n[len(n) // 2], n_now=len(n),
                      lo=p[int(0.05 * (len(p) - 1))] if p else None,
                      hi=p[int(0.95 * (len(p) - 1))] if p else None, n_pre=len(p))
    return out


# ------------------------------------------------------------------ display forms
def D_between(lo, hi, lo_name, hi_name):
    """A percentage of the way between two named ends."""
    span = float(hi) - float(lo)
    return dict(kind="between", a=100.0 / span, b=-100.0 * float(lo) / span, unit="%",
                lo_name=lo_name, hi_name=hi_name)


def D_per100(noun):
    """A share as a count per 100 of `noun`."""
    return dict(kind="per100", a=100.0, b=0.0, unit="%", noun=noun)


def D_auc(noun):
    """A ranking probability as a count of pairs per 100."""
    return dict(kind="auc", a=100.0, b=0.0, unit="%", noun=noun)


def D_unit(name, mul=1.0, short=None):
    tick = short or name
    return dict(kind="unit", a=float(mul), b=0.0, unit=("" if tick[:1] in ("%", "°") else " ") + tick,
                unit_long=name)


def D_count(noun, per_one=None, per_many=None):
    """A count, e.g. 1.6 pairs per body; under 1 it is said per 100 (40 pairs per 100 bodies)."""
    return dict(kind="count", a=1.0, b=0.0, unit="", noun=noun, per_one=per_one, per_many=per_many)


def D_change():
    """A signed fraction of change, shown in percent (+0.6% = fell by 0.6% of its value)."""
    return dict(kind="change", a=100.0, b=0.0, unit="%")


def shown(display, value):
    """The display value (what the chart plots) for a raw value."""
    return None if value is None else display["a"] * float(value) + display["b"]


def say(spec, value, full=True):
    """A raw value in its display form. full=False gives the number with its unit only."""
    d = spec["display"]
    if value is None:
        return "unmeasured"
    v = shown(d, value)
    kind = d["kind"]
    if kind == "between":
        text = fixed(v, 2) + "%"
        return text + (f" of the way from '{d['lo_name']}' to '{d['hi_name']}'" if full else "")
    if kind in ("per100", "auc"):
        return per(float(value), d["noun"] if full else "")
    if kind == "change":
        return ("+" if v > 0 else "") + fixed(v, 2) + "%"
    if kind == "count":
        if 0 < abs(v) < 1:
            many = d.get("per_many") or "windows"
            return f"{fixed(100 * v, 2)} {d['noun']} per 100 {many}"
        tail = (" " + d["noun"] + (" per " + d["per_one"] if d.get("per_one") else "")) if full else ""
        return fixed(v, 2) + tail
    name = d["unit_long"]
    return fixed(v, 3 if abs(v) < 10 else 2) + ("" if name.startswith("%") else " ") + name


def say_ref(spec, value):
    """A reference value (floor, goal, end) in the display form, number and unit only."""
    return say(spec, value, full=False)


# ------------------------------------------------------------------ the registry
# Every entry: how to read it from one `mimo` record, a plain title, the two ends of its scale in
# lived terms (value, meaning), its display form, and where known its floor (what knowing nothing
# scores) and goal line (what the number reads when the capability works). goal None is drawn and
# written as "no validated success target" with `goal_why` when a reason is known. `block`/`flag` name the
# DESIGN.md section 3 block and the launch flag that write it, so an absent value can say why.
SERIES: dict = {}


def _s(key, get, title, lo, hi, display, *, floor=None, goal=None, better="higher", block=None,
       flag=None, floor_series=None, limit_series=None, refs=None, band=None, note=None,
       goal_why=None, field=None, seam=None):
    SERIES[key] = dict(key=key, get=get, title=title, lo=list(lo), hi=list(hi) if hi else None,
                       display=display, floor=list(floor) if floor else None,
                       goal=list(goal) if goal else None, better=better, block=block, flag=flag,
                       floor_series=floor_series, limit_series=limit_series,
                       refs=[list(r) for r in refs] if refs else None,
                       band=list(band) if band else None, note=note, goal_why=goal_why,
                       field=field, seam=dict(seam) if seam else None)


FOLLOW_ENDS = ((0, "the turn is unrelated to the command"), (1, "exactly as commanded"))
FOLLOW_DISPLAY = D_between(0, 1, "the turn is unrelated to the command", "exactly as commanded")
_FOLLOW_LINES = follow_lines(FOLLOW)
FOLLOW_REFS = _FOLLOW_LINES["refs"]
FOLLOW_GOAL = _FOLLOW_LINES["goal"]
FOLLOW_GOAL_WHY = _FOLLOW_LINES["goal_why"]
FOLLOW_BOUT_REFS = _FOLLOW_LINES["bout_refs"]
FOLLOW_BOUT_GOAL = _FOLLOW_LINES["bout_goal"]
FOLLOW_BOUT_GOAL_WHY = _FOLLOW_LINES["bout_goal_why"]
FOLLOW_TURNS = independent_turns_text(FOLLOW)

# ------------------------------------------------------------------ definition seams
# A gauge whose definition changed at a commit: the readings from the first segment whose code
# contains that commit are a different measurement from the readings before it, so the pages say
# where that segment begins (seam_text) instead of joining the two into one line with no mark.
# Commit b55ee9a4d (2026-09-23, docs/gain_of_function_v3/agents/fix-following-gauge.md) changed the
# world's following gauge: env.py took the LAST handle row for the eyes, which under trunk_v1 (every
# H84 segment) is the chest. The before/after readings are measurements/following_gauge_before_after
# .json: 400 seeded orders in segment 31's world, the old and the new gauge side by side.
_GAUGE_FIX = dict(commit="b55ee9a4d", date="2026-09-23",
                  source="docs/gain_of_function_v3/measurements/following_gauge_before_after.json")
SEAM_HELD = dict(_GAUGE_FIX, change=(
    "before it, this gauge scored the eyes' row in place of the chest, read each command over 26 frames (260 ms) "
    "and averaged a reading from every model step, most of them running into the next command; from it, the chest "
    "is scored, the eyes' row is not, and each command is read once over exactly its own 25 frames (250 ms). On the "
    "same 400 seeded commands the old gauge read 14% of the way from 'the turn is unrelated to the command' to "
    "'exactly as commanded' and the new one 30%"))
SEAM_FRAME = dict(_GAUGE_FIX, change=(
    "before it, this one-frame (10 ms) reading scored the eyes' row in place of the chest; from it, the chest is "
    "scored and the eyes' row is not. On the same 400 seeded commands the old reading was 6.7% of the way from "
    "'the turn is unrelated to the command' to 'exactly as commanded' and the new one 7.0%"))
SEAM_ASK = dict(_GAUGE_FIX, change=(
    "before it, the size of the turn asked counted the eyes' row, whose head-relative eye direction set against "
    "the eye body's turn in the world read as a 161-degree ask, and left the chest out; from it, the chest is "
    "counted and the eyes' row is not. On the same 400 seeded commands the average ask was 23 degrees before and "
    "17 degrees after"))

# --- already logged on every H84 segment
for _sense, _name in ALL_SENSES:
    _s(f"act.{_sense}", _field("prediction_audit", "modalities", _sense, "improvement_over_shuffled"),
       f"{_name.capitalize()}: gain from knowing its own executed command (against another body's absolute command)",
       (0, "the command is not used: another body's absolute command predicts as well"), (1, "exact prediction"),
       D_between(0, 1, "the command is not used", "exact prediction"),
       floor=(0, "another body's absolute command would do as well"),
       note="the original control (another body's command as it stands), kept for one segment beside the "
            "size-matched control (act_sized) from the code of 2026-10-02, then retired",
       field=f"prediction_audit.modalities.{_sense}.improvement_over_shuffled")
    _s(f"pers.{_sense}", _field("prediction_audit", "modalities", _sense, "improvement_over_persistence"),
       f"{_name.capitalize()}: next model step predicted better than \"nothing changes\"",
       (0, "no better than predicting that nothing changes"), (1, "exact prediction"),
       D_between(0, 1, "no better than 'nothing changes'", "exact"),
       floor=(0, "predicting that nothing changes"),
       field=f"prediction_audit.modalities.{_sense}.improvement_over_persistence")
    # b1 computes it on every audit step with more than one body, flags on or off (reviews/
    # b1-causal-efference.md MINOR 10): no flag gates it
    _s(f"sens.{_sense}", _field("prediction_audit", "modalities", _sense, "sensitivity"),
       f"{_name.capitalize()}: how far another body's absolute command moves the prediction",
       (0, "changing the command changes nothing"),
       (1, "changing the command moves the prediction as far as the predicted change itself"),
       D_between(0, 1, "changing the command changes nothing",
                 "the command moves the prediction as far as the predicted change itself"),
       block="prediction_audit", note="measured only when more than one body trains at once",
       field=f"prediction_audit.modalities.{_sense}.sensitivity")

# --- the swap test's size-matched control and its second phase (2026-10-02, mimo_bridge/prediction_audit.py
# DEFINITIONS; docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md): logged from the first segment on
# that code, so a record before it reads unmeasured, never 0
for _sense, _name in ALL_SENSES:
    _s(f"act_sized.{_sense}", _field("prediction_audit", "modalities", _sense, "improvement_over_size_matched"),
       f"{_name.capitalize()}: gain from knowing its own executed command (against another body's size-matched "
       f"command)",
       (0, "the command is not used: another body's size-matched command predicts as well"), (1, "exact prediction"),
       D_between(0, 1, "the command is not used", "exact prediction"),
       floor=(0, "another body's size-matched command would do as well"),
       note="the control is another body's asked moves and turns, measured from its own pose, applied to this "
            "body's pose; graded on model steps that are multiples of 20 (with --motor-hold 5, the last model step "
            "of a hold)",
       field=f"prediction_audit.modalities.{_sense}.improvement_over_size_matched")
    _s(f"act_first.{_sense}", _field("prediction_audit_hold_first", "senses", _sense, "all_bodies",
                                     "improvement_over_size_matched"),
       f"{_name.capitalize()}: gain from knowing its own executed command, on a hold's first model step",
       (0, "the command is not used: another body's size-matched command predicts as well"), (1, "exact prediction"),
       D_between(0, 1, "the command is not used", "exact prediction"),
       floor=(0, "another body's size-matched command would do as well"),
       note="graded on the first model step of each hold (the command is new; the context still holds the "
            "previous one), against another body's size-matched command",
       field=f"prediction_audit_hold_first.senses.{_sense}.all_bodies.improvement_over_size_matched")

_s("follow", _field("cmd_response_cos_held"), "The body turns the way it was commanded (every command)",
   (-1, "the body turns against the command"), (1, "exactly as commanded"), FOLLOW_DISPLAY,
   floor=(0, "the turn is unrelated to the command"), goal=FOLLOW_GOAL, refs=FOLLOW_REFS,
   goal_why=FOLLOW_GOAL_WHY,
   note="the direction match over each command's 250 ms life, averaged; size is not scored; not yet "
        "measured as a count per window",
   field="cmd_response_cos_held", seam=SEAM_HELD)
_s("follow_bout", _field("follow_bout"), "The body turns the way it was commanded, movement bouts only",
   (-1, "against the command"), (1, "exactly as commanded"), FOLLOW_DISPLAY,
   floor=(0, "unrelated to the command"), goal=FOLLOW_BOUT_GOAL, refs=FOLLOW_BOUT_REFS,
   goal_why=FOLLOW_BOUT_GOAL_WHY,
   note="the direction match over one 10 ms frame, the last frame of each model step during movement "
        "bouts, as a running average; not yet measured as a count per window",
   field="follow_bout", seam=SEAM_FRAME)
_s("cmd_deg", lambda r: (lambda v: None if v is None else math.degrees(v))(_num(r.get("cmd_mag_rad"))),
   "Size of the turn each command asks for", (0, "no turn asked"), None, D_unit("degrees", short="°"),
   field="cmd_mag_rad", seam=SEAM_ASK)
_s("planner_share", _field("exec_source", "planner"), "Commands written by the planner",
   (0, "babble and reflexes only"), (1, "every command"), D_per100("latched commands"),
   field="exec_source.planner")
_s("gap_closed", _field("intention", "gap_closed_pct"), "How much of the way to its chosen goal a plan closes",
   (0, "never moved toward it"), (100, "arrived"), D_unit("% of the way to its goal", short="%"),
   field="intention.gap_closed_pct")
_s("consequence_goal", _field("consequence", "goal_rate"), "Chosen goals that carried a consequence",
   (0, "never"), (1, "every chosen goal"), D_per100("chosen goals"), field="consequence.goal_rate")
_s("rank1", _field("tenure_tok", "rank1"), "Picks its own command's effect out of a lineup",
   (0, "never"), (1, "every time"), D_per100("lineups"), floor_series="rank1_chance",
   field="tenure_tok.rank1")
_s("rank1_chance", lambda r: (lambda n: None if n is None else 1.0 / (1.0 + n))(_num(_g(r, "tenure_tok", "n_decoys"))),
   "Chance of picking it by luck", (0, "never"), (1, "every time"), D_per100("lineups"),
   field="tenure_tok.n_decoys")
_s("tenure_margin", lambda r: (lambda g, p: None if g is None or not p else 100.0 * g / p)(
    _num(_g(r, "tenure_gap", "pose")), _num(_g(r, "tenure_pe", "pose"))),
   "Command-aware 250 ms pose prediction beats do-nothing by", (0, "no better than do-nothing"),
   (100, "exact"), D_unit("% better than doing nothing", short="%"),
   note="the tenure pose head had 0 of its 256 units live on ckpt_step_344400 (DESIGN.md section 10)",
   field="tenure_gap.pose / tenure_pe.pose")
for _depth in range(3):
    _s(f"plan_license.{_depth + 1}", _list_item(("plan_license", "win_emas"), _depth),
       f"Imagined {_depth + 1}-step prediction beats doing nothing", (0, "never"), (1, "always"),
       D_per100("tries"), floor=(0.5, "the line where a lookahead depth is licensed"),
       field=f"plan_license.win_emas[{_depth}]")
# WHAT THE COHERENCE CHANNEL'S INPUT IS (the operator's decision of Oct 3, 2026: named for what it measures now,
# replaced after the motor-encoder repair). Under --wellness-coherence extent the channel carries the body map's
# extent (lem/model/body_schema.py self_extent): the mean of one co-movement number per handle and per outside
# thing. The handle statistic has no time-lag bank, no sign and no passive-movement comparison, so it counts
# co-movement and not causation (docs/gain_of_function_v4/coordinator/own_command_analysis_2026-10-02.html;
# tests/test_body_schema_co_movement.py holds its readings on synthetic inputs).
CO_MOVEMENT_NOTE = ("the body map's co-movement statistic: per handle, a running average of (size of the "
                    "command's change x size of each proprioception number's change), each size divided by its own "
                    "running root-mean-square, passed through sigmoid(5 x (mean - one half)); it counts changes that "
                    "fall in the same model step and does not measure whether the command caused them (no time-lag "
                    "bank, no sign, no comparison against movement without a command)")
for _k in ("prediction_error", "tactile_comfort", "capability", "coherence", "motor_state",
           "self_touch", "movement"):
    _s(f"wellness.{_k}", _field("wellness", _k),
       "Wellness channel: coherence (co-movement of command and proprioception)" if _k == "coherence"
       else f"Wellness channel: {_k.replace('_', ' ')}",
       (0, "its lowest"), (1, "its highest"), D_between(0, 1, "its lowest", "its highest"),
       note=("window mean over the bodies; with the body map on, its input is the mean over the 26 handles and "
             "the outside things of " + CO_MOVEMENT_NOTE if _k == "coherence" else "window mean over the bodies"),
       field=f"wellness.{_k}")
_s("extent", _field("incorporation", "extent"), "Body map: co-movement of command and proprioception (extent)",
   (0.0759, "the map's real low end (no co-movement recorded)"),
   (0.9241, "the map's real high end (the two change sizes matched on every model step)"),
   D_between(0.0759, 0.9241, "the map's real low end", "its real high end"),
   note="the mean over the 26 handles and the outside things the map watches; a handle's number is "
        + CO_MOVEMENT_NOTE, field="incorporation.extent")
_s("self_pairs", _field("self_contact_pairs"), "Pairs of its own body parts touching, per body",
   (0, "no self-touch recorded in this reporting interval"), None, D_count("pairs", "body", "bodies"), note="window mean over the bodies",
   field="self_contact_pairs")
_s("self_force_share", _field("self_contact_force_share"), "Touch force that is self-touch",
   (0, "none"), (1, "all of it"), D_per100("parts of the touch force"), field="self_contact_force_share")
_s("self_touch_act", _field("self_touch_controlled", "improvement_over_shuffled"),
   "Tactile prediction for bodies touching themselves",
   (0, "the same error as using another body's command"), (1, "exact prediction"),
   D_between(0, 1, "the same error as using another body's command", "exact prediction"),
   note="Scores all tactile information for bodies touching themselves, including floor contact; it does not isolate self-touch",
   floor=(0, "a shuffled command would do as well"), field="self_touch_controlled.improvement_over_shuffled")
_s("posture.mean", lambda r: _mean(_rolls(r)), "Average turn away from lying on the back",
   (0, "on its back"), (180, "belly down"), D_unit("degrees", short="°"),
   note="90 degrees = on its side; mean over the bodies at the last model step of the window",
   field="body_orientation[].roll_degrees")
_s("posture.max", lambda r: (lambda v: max(v) if v else None)(_rolls(r)), "Most-turned body",
   (0, "on its back"), (180, "belly down"), D_unit("degrees", short="°"), note="90 degrees = on its side",
   field="body_orientation[].roll_degrees")
_s("hand_in_view", _field("task_hand_fov"), "Model steps a hand is in the sharp centre of view",
   (0, "never"), (1, "always"), D_per100("model steps"), note="exposure; not a skill score",
   field="task_hand_fov")
_s("mobile_hand", _field("task_mobile_hand"), "Model steps a hand touches the mobile",
   (0, "never"), (1, "always"), D_per100("model steps"), field="task_mobile_hand")
_s("self_forecast_1", _field("self_state", "forecast_vs_persistence_1"),
   "Own-body forecast one step ahead, against \"nothing changes\"",
   (1, "no better than 'nothing changes'"), (0, "exact"),
   D_between(1, 0, "no better than 'nothing changes'", "exact"), field="self_state.forecast_vs_persistence_1")
_s("sps", _field("model_steps_per_s"), "Model steps per second", (0, "stalled"), None,
   D_unit("model steps per second", short="steps/s"), field="model_steps_per_s")
_s("vram", _field("peak_vram_mb"), "Card memory the trainer allocated at its peak", (0, "none"), None,
   D_unit("GiB", mul=1 / 1024.0), field="peak_vram_mb")

# --- DESIGN.md section 3 blocks (present only while their flag is on)
GOAL_WHY_CHAIN = "the planner does not aim at predicted wellness (it descends on distance to a remembered model step)"
_s("cc.chosen_best", _field("causal_chain", "chosen_best_share"),
   "Plans whose chosen action was the one it predicted would feel best",
   (0, "never"), (1, "every plan"), D_per100("plans"),
   block="causal_chain", flag="causal_chain", goal_why=GOAL_WHY_CHAIN,
   note="counted only when the gap to the best alternative exceeds the evaluator's tie band, when the "
        "record carries one (chosen_tied_share); older records count ties as best",
   field="causal_chain.chosen_best_share")
_s("cc.chance", lambda r: (lambda n: None if n is None else 1.0 / (1.0 + n))(_num(_g(r, "causal_chain", "alternatives_n"))),
   "Theoretical random-choice rate, only if exactly one option is distinguishably best", (0, "never"), (1, "every plan"), D_per100("plans"),
   block="causal_chain", flag="causal_chain", field="causal_chain.alternatives_n",
   note="not a measured comparison for this window; the formula assumes one distinguishable winner and cannot account for ties")
_s("cc.tied", _field("causal_chain", "chosen_tied_share"),
   "Plans tied: predicted lead inside the evaluator's own error band",
   (0, "never"), (1, "every plan"), D_per100("plans"), block="causal_chain", flag="causal_chain",
   note="a tie is a predicted difference within the evaluator's own error band, which has been wider than the "
        "candidates' predicted spread in every window since Sep 23 (median 12.6 times the spread), so ties are certain "
        "whatever the forward model predicts: no evidence about the forward model. Its planner-side reading is the "
        "planner recheck's captured share",
   field="causal_chain.chosen_tied_share")


def tie_meaning(cc):
    """What a planner tie is, with this window's tie band against the spread of the candidates' predicted wellness
    (the own-command analysis review of 2026-10-02), for the sentence beside a tied share. A choice is tied when the
    chosen command's predicted wellness sum lies within the band of the best alternative's (mimo_bridge/
    causal_chain.py outcome); the band is the evaluator's own error on a difference between two candidates
    (TIE_BAND). While the band is wider than the spread of the five candidates' predicted sums
    (candidate_spread_mean), every choice reads tied whatever the forward model predicts: on H84 it was wider in
    1,174 of 1,174 windows since Sep 23, median 12.6 times the spread (docs/gain_of_function_v4/agents/
    hold_loss_history_2026-10-02.json). So the tied share is no evidence about the forward model; the planner
    recheck's captured share is its planner-side reading."""
    def num(v):
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
    band, spread = num((cc or {}).get("tie_band")), num((cc or {}).get("candidate_spread_mean"))
    text = (" (a tie: the chosen command's predicted wellness sum lies within the tie band of the best alternative's;"
            " the band is the evaluator's own error on a difference between two candidates"
            + (f", {pct(band / 5.0, 2)} of the 0-to-5 wellness scale" if band is not None else ""))
    if band and spread and spread > 0:
        text += (f", {fixed(band / spread, 2)} times the spread predicted between the candidates in this window, so"
                 " every choice reads tied whatever the forward model predicts" if band >= spread else
                 ", narrower than the spread predicted between the candidates in this window")
    return text + ("; the tied share is no evidence about the forward model: its planner-side reading is the planner"
                   " recheck's captured share on the Development page)")
_WELL5 = D_between(0, 5, "no difference", "the whole 0-to-5 range of predicted wellness")
_s("cc.gap", _field("causal_chain", "gap_to_best_alt_mean"),
   "Chosen movement’s predicted condition minus the best alternative (negative means worse)", (0, "no difference"),
   (5, "the whole 0-to-5 range"), _WELL5, block="causal_chain", flag="causal_chain",
   field="causal_chain.gap_to_best_alt_mean")
_s("cc.spread", _field("causal_chain", "candidate_spread_mean"),
   "Spread of predicted wellness over the candidates", (0, "all the same"), (5, "the whole 0-to-5 range"),
   _WELL5, block="causal_chain", flag="causal_chain", field="causal_chain.candidate_spread_mean")
_s("cc.tie_band", _field("causal_chain", "tie_band"),
   "The evaluator's tie band (differences it cannot resolve)", (0, "resolves every difference"),
   (5, "the whole 0-to-5 range"), _WELL5, block="causal_chain", flag="causal_chain",
   field="causal_chain.tie_band")
_s("cc.wellness_corr", _field("causal_chain", "wellness_change_corr"),
   "Predicted wellness change matches the change that followed",
   (-1, "the opposite of what happened"), (1, "exactly what happened"),
   D_between(0, 1, "unrelated to what happened", "exactly what happened"),
   floor=(0, "unrelated to what happened"), block="causal_chain", flag="causal_chain",
   note="a correlation over the window's commands, shown as a share of the way; not yet measured as a count",
   field="causal_chain.wellness_change_corr")
_s("cc.reached_body", _field("causal_chain", "reached_body_share"),
   "Plans whose first command reached the body", (0, "none"), (1, "every plan"), D_per100("plans"),
   block="causal_chain", flag="causal_chain", field="causal_chain.reached_body_share")
# the order the body received against the one the planner imagined sending (reviews/
# b1-causal-efference.md MINOR 2): planner_share_of_executed was the smoother's setting, 0.35 by
# construction, and is no longer shown. With the motor interface's limb fixes on, the order received
# is the one the worlds recorded after projecting it (causal_chain.on_executed)
_s("cc.plan_exec_cm", _field("causal_chain", "plan_exec_distance_cm"),
   "Distance between the command the planner imagined and the command the body received",
   (0, "the body got what was imagined"), None, D_unit("cm"), better="lower",
   block="causal_chain", flag="causal_chain", field="causal_chain.plan_exec_distance_cm")
for _sense, _name in SENSES:
    _s(f"cc.next.{_sense}", _ratio_gain(("causal_chain", "pred_err_next", _sense), ("causal_chain", "persist_err_next", _sense)),
       f"{_name.capitalize()}: prediction made before the command, better than \"nothing changes\"",
       (0, "no better than 'nothing changes'"), (1, "exact"),
       D_between(0, 1, "no better than 'nothing changes'", "exact"), floor=(0, "\"nothing changes\""),
       block="causal_chain", flag="causal_chain",
       field=f"causal_chain.pred_err_next.{_sense} / persist_err_next.{_sense}")
_s("eff.body", _ratio_gain(("efference", "body_cm_err_latch"), ("efference", "body_cm_do_nothing_latch")),
   "Where each body part goes over a command: error removed against predicting no movement",
   (0, "no better than predicting no movement"), (1, "exact to the centimetre"),
   D_between(0, 1, "no better than predicting no movement", "exact to the centimetre"),
   floor=(0, "predicting no movement"), floor_series="eff.body_shuffled",
   block="efference", flag="efference_heads", field="efference.body_cm_err_latch / body_cm_do_nothing_latch")
_s("eff.body_shuffled", _ratio_gain(("efference", "body_cm_shuffled_latch"), ("efference", "body_cm_do_nothing_latch")),
   "The same with another body's command (control)", (0, "no better than predicting no movement"),
   (1, "exact"), D_between(0, 1, "no better than predicting no movement", "exact to the centimetre"),
   block="efference", flag="efference_heads", field="efference.body_cm_shuffled_latch")
_s("eff.body_cm", _field("efference", "body_cm_err_latch"), "Error in where each body part goes over a command",
   (0, "exact"), None, D_unit("cm"), better="lower", block="efference", flag="efference_heads",
   field="efference.body_cm_err_latch")
for _h in range(3):
    _s(f"eff.gaze.{_h + 1}", lambda r, h=_h: (lambda e, b: None if e is None or not b else 1.0 - e / b)(
        _list_item(("efference", "gaze_deg_err"), h)(r), _list_item(("efference", "gaze_deg_do_nothing"), h)(r)),
       f"Where the eyes point {50 * (_h + 1)} ms later: error removed against predicting the eyes stay",
       (0, "no better than predicting the eyes stay"), (1, "exact to the degree"),
       D_between(0, 1, "no better than predicting the eyes stay", "exact to the degree"),
       floor=(0, "predicting the eyes stay"), block="efference", flag="efference_heads",
       note="eye angles only; what the eyes will see is the gaze probe's reading (item 7)",
       field=f"efference.gaze_deg_err[{_h}] / gaze_deg_do_nothing[{_h}]")
_CTL = D_between(0, 1, "unrelated to what it expected", "exactly as expected")
_s("ctl.agreement", _field("controllability", "agreement_mean"),
   "What the body did matches what it expected its command to do",
   (-1, "the opposite of what it expected"), (1, "exactly as expected"), _CTL,
   floor_series="ctl.control", block="controllability", flag="controllability",
   note="an agreement of direction and size, averaged; not yet measured as a count",
   field="controllability.agreement_mean")
_s("ctl.control", _field("controllability", "control_mean"),
   "The same body's result against another body's command (control)",
   (-1, "the opposite"), (1, "exact"), _CTL, block="controllability", flag="controllability",
   field="controllability.control_mean")
_s("ctl.difference", _field("controllability", "difference_mean"),
   "Agreement with its own command minus agreement with another body's command",
   (-1, "the other command agrees more"), (1, "only its own command agrees"), _CTL,
   block="controllability", flag="controllability", field="controllability.difference_mean")
_s("ctl.value", _field("controllability", "value_mean"), "Controllability as wellness receives it",
   (-1, "the opposite of what it expected"), (1, "full control"), _CTL,
   block="controllability", flag="controllability", field="controllability.value_mean")
# --- controllability's size-matched control (2026-10-03, lem/model/wellness_channels.py SHADOW_CONTROL_KIND;
# gate_definitions in the stats block): recorded only, logged from the first segment on that code, so a record
# before it reads unmeasured, never 0. It feeds the shadow gate (controllability_gates_text below)
_s("ctl.control_sized", _field("controllability", "size_matched_control_mean"),
   "The same body's result against another body's size-matched command (control, recorded only)",
   (-1, "the opposite"), (1, "exact"), _CTL, block="controllability", flag="controllability",
   note="the body head's expectation formed from another body's asked moves and turns, measured from that body's "
        "own pose and applied to this body's pose, against what this body did over the command's hold (5 model "
        "steps on H84); it feeds the shadow gate, which no weight reads",
   field="controllability.size_matched_control_mean")
_s("ctl.difference_sized", _field("controllability", "size_matched_difference_mean"),
   "Agreement with its own command minus agreement with another body's size-matched command",
   (-1, "the other command agrees more"), (1, "only its own command agrees"), _CTL,
   block="controllability", flag="controllability",
   note="recorded only: one reading of the shadow gate is this difference over the bodies measured on one graded "
        "command; the goal ranking's controllability weight reads the gate fed by another body's absolute command",
   field="controllability.size_matched_difference_mean")


def controllability_gates_text(b):
    """Both controllability gates on the same graded commands, as sentences of counts for either page, or None
    when the block carries no shadow gate (a record written before the code of 2026-10-03).

    Input: the stats record's controllability block (lem/model/wellness_channels.py ControllabilityTracker.summary;
    gate_definitions in the block defines every key). The gate in force takes another body's absolute command as
    its control (that body's targets as they stand) and sets the controllability weight in the goal ranking; the
    shadow gate takes another body's size-matched command (its asked moves and turns, measured from its own pose,
    applied to this body's pose) and is recorded only. Each takes one reading per graded command (the bodies'
    commands latched on one model step are graded together when the next commands appear), opens when the mean of
    its last 32 readings is above 3 of their standard errors and shuts under 1. Output: per report window, the
    graded commands each gate judged and those after which it stood open (gate_commands, gate_open_commands,
    shadow_gate_commands, shadow_gate_open_commands), the split on the commands both judged (gates_same_commands,
    gates_both_open_commands, gate_only_open_commands, shadow_gate_only_open_commands) and each gate's state at
    the window's end (gate_open, shadow_gate_open)."""
    if not isinstance(b, dict) or _num(b.get("shadow_gate_commands")) is None:
        return None

    def count(key):
        return int(_num(b.get(key)) or 0)

    def state(key):
        return "open" if b.get(key) else "shut"
    def judged_text(name, control, commands, opened):
        if not commands:
            return f"{name} ({control}) took no reading"
        noun = "graded command" if commands == 1 else "graded commands"
        return f"{name} ({control}) stood open after {opened:,} of the {commands:,} {noun} it judged"
    judged, shadow_judged = count("gate_commands"), count("shadow_gate_commands")
    if not judged and not shadow_judged:
        text = ("Neither controllability gate took a reading in this report window: no graded command had a body "
                "whose expected and commanded effects were above the stillness floor")
    else:
        text = ("In this report window "
                + judged_text("the gate in force", "its control is another body's absolute command", judged,
                              count("gate_open_commands"))
                + ", and "
                + judged_text("the shadow gate", "its control is another body's size-matched command; recorded "
                              "only, no weight reads it", shadow_judged, count("shadow_gate_open_commands")))
        if judged and not shadow_judged:
            text += (": no graded command carried the body head's expectation from a size-matched command (it "
                     "needs the efference heads and two or more bodies)")
        same = count("gates_same_commands")
        if same:
            text += (f". On the {same:,} graded commands both gates judged, both stood open after "
                     f"{count('gates_both_open_commands'):,}, only the gate in force after "
                     f"{count('gate_only_open_commands'):,} and only the shadow gate after "
                     f"{count('shadow_gate_only_open_commands'):,}")
    return (text + f". At the window's end the gate in force is {state('gate_open')} and the shadow gate is "
                   f"{state('shadow_gate_open')}.")


_s("ctl.measured", _field("controllability", "measured_share"),
   "Commands scored (an effect expected and commanded above the floor)", (0, "none"), (1, "every command"),
   D_per100("commands"), block="controllability", flag="controllability",
   field="controllability.measured_share")
_s("lp.value", _field("learning_progress", "value_mean"), "Learning progress: prediction errors falling",
   (-1, "errors doubled"), (1, "errors gone"), D_change(), floor_series="lp.control",
   band=(LP_FROZEN_BAND[0], LP_FROZEN_BAND[1], "frozen weights, where no learning can happen"),
   block="learning_progress", flag="learning_progress",
   note="the fall of model error as a share of 'nothing changes' error, recent against long-run; "
        "negative = getting worse",
   field="learning_progress.value_mean")
_s("lp.control", _field("learning_progress", "control_mean"), "The same on a same-model-step control",
   (-1, "errors doubled"), (1, "errors gone"), D_change(), block="learning_progress",
   flag="learning_progress", field="learning_progress.control_mean")
for _sense, _name in ALL_SENSES:
    _s(f"lp.{_sense}", _field("learning_progress", "per_sense", _sense), f"Learning progress, {_name}",
       (-1, "errors doubled"), (1, "errors gone"), D_change(),
       block="learning_progress", flag="learning_progress", field=f"learning_progress.per_sense.{_sense}")
_s("wr.lead", lambda r: (lambda lc: None if lc is None else lc[1])(lead_channel(r)),
   "How much the strongest internal score contributes to the chosen goal",
   (0, "no part of the lead"), (1, "the whole lead"), D_per100("parts of the lead"),
   block="wellness_rank", flag="wellness_rank",
   note="the channels' shares add to 100 and can be below 0 (a channel pushing against the winner) or "
        "above 100 (pushing for it by more than the whole lead)",
   field="wellness_rank.winner_margin_share")
_s("wr.lead_weight", lambda r: (lambda lc: None if lc is None else lc[2])(lead_channel(r)),
   "The same channel's share of the ranking weights", (0, "no weight"), (1, "all the weight"),
   D_per100("parts of the weight"), block="wellness_rank", flag="wellness_rank",
   field="wellness_rank.weight_share_by_channel (else weights)")
_s("wr.age_mean", _field("wellness_rank", "goal_age", "mean_share_back"),
   "How old the chosen goal is among the remembered states", (0, "the newest remembered model step"),
   (1, "the oldest, 10 s back"), D_between(0, 1, "the newest remembered model step", "the oldest (10 s back)"),
   floor=(0.5, "a choice blind to age sits half-way back on average"),
   block="wellness_rank", flag="wellness_rank", field="wellness_rank.goal_age.mean_share_back")
_s("wr.age_oldest", _field("wellness_rank", "goal_age", "oldest_tenth_share"),
   "Goals chosen from the oldest tenth of the remembered states", (0, "none"), (1, "every goal"),
   D_per100("goals"), floor=(0.1, "a choice blind to age puts 10 of every 100 in each tenth"),
   block="wellness_rank", flag="wellness_rank", field="wellness_rank.goal_age.oldest_tenth_share")
_s("wr.age_newest", _field("wellness_rank", "goal_age", "newest_tenth_share"),
   "Goals chosen from the newest tenth of the remembered states", (0, "none"), (1, "every goal"),
   D_per100("goals"), floor=(0.1, "a choice blind to age puts 10 of every 100 in each tenth"),
   block="wellness_rank", flag="wellness_rank", field="wellness_rank.goal_age.newest_tenth_share")
_s("wr.gated", _field("wellness_rank", "integrity_gated_share"), "Goal candidates barred by the integrity gate",
   (0, "none"), (1, "all"), D_per100("candidate model steps"), block="wellness_rank", flag="wellness_rank",
   field="wellness_rank.integrity_gated_share")
_s("hp.self_share", _field("hand_press", "self_share"), "Model steps a hand presses its own body",
   (0, "none"), (1, "every model step"), D_per100("model steps"), block="hand_press", field="hand_press.self_share")
_s("hp.world_share", _field("hand_press", "world_share"), "Model steps a hand presses the world",
   (0, "none"), (1, "every model step"), D_per100("model steps"), block="hand_press", field="hand_press.world_share")
_ONSET = D_auc("pairs of skin readings (one new touch and one no-touch)")
_s("ccx.onset_self", _field("contact_control", "onset_auc_self"),
   "Own-body touch foreseen before it begins",
   (0.5, "a coin flip"), (1, "every onset ranked above every non-onset"), _ONSET,
   floor=(0.5, "a coin flip"), floor_series="ccx.onset_self_control", block="contact_control",
   flag="contact_head",
   note="ranked over patches that were untouched the model step before: in how many (touched, untouched) "
        "pairs the patch about to be touched got the higher chance",
   field="contact_control.onset_auc_self")
_s("ccx.onset_self_control", _field("contact_control", "control_onset_auc_self"),
   "A base-rate guess on the same skin areas at the same times (control)", (0.5, "a coin flip"), (1, "every onset"),
   _ONSET, block="contact_control", flag="contact_head", field="contact_control.control_onset_auc_self")
_s("ccx.onset_world", _field("contact_control", "onset_auc_world"),
   "Touch by the world foreseen before it begins", (0.5, "a coin flip"), (1, "every onset"), _ONSET,
   floor=(0.5, "a coin flip"), floor_series="ccx.onset_world_control", block="contact_control",
   flag="contact_head", field="contact_control.onset_auc_world")
_s("ccx.onset_world_control", _field("contact_control", "control_onset_auc_world"),
   "A base-rate guess on the same skin areas at the same times (control)", (0.5, "a coin flip"), (1, "every onset"),
   _ONSET, block="contact_control", flag="contact_head", field="contact_control.control_onset_auc_world")
_s("ccx.paired", _field("contact_control", "success_paired"),
   "New self-touches predicted with its own command", (0, "none"), (1, "every onset"),
   D_per100("new self-touches"), floor_series="ccx.success_control", block="contact_control",
   flag="contact_head", note="the same onsets as the control beside it",
   field="contact_control.success_paired")
_s("ccx.success_control", _field("contact_control", "success_control"),
   "The same onsets with another body's command (control)", (0, "none"), (1, "every onset"),
   D_per100("new self-touches"), block="contact_control", flag="contact_head",
   field="contact_control.success_control")
_s("ccx.auc", _field("contact_control", "head_auc"),
   "Any-contact score over every patch at every model step (persisting contacts included)",
   (0.5, "a coin flip"), (1, "every touch ranked above every non-touch"), _ONSET,
   floor=(0.5, "a coin flip"), block="contact_control", flag="contact_head",
   note="mostly measures whether a touch that is already there continues", field="contact_control.head_auc")
_s("ccx.habituation", _field("contact_control", "habituation_mean"),
   "Average pay factor for foreseen touches", (0, "a pair repeated without end"), (1, "a new pair"),
   D_between(0, 1, "a pair repeated without end", "a new pair"), block="contact_control",
   flag="contact_head", note="the pay factor is 1/(1 + repeats): 100% = a new pair, lower = repeated",
   field="contact_control.habituation_mean")
_s("sce.onsets", _field("self_contact_events", "onsets"), "Self-touches that began in the window",
   (0, "none"), None, D_count("self-touches", "window", "windows"), block="self_contact_events",
   field="self_contact_events.onsets")


def _synergy_following(r):
    """How well the body follows synergy-written orders: the synergy's (or the planner's, which the
    synergy writes) entry of synergy.following_by_author, else None. b5's following_mean is every
    graded order's direction match, babble and reflex orders included (lem_model.py _synergy_grade),
    so it is shown as a stand-in (syn.following_all) and never as this item's measurement
    (verify-fix-dashboards 2026-09-23; field contract: "following by author / b5's physics check")."""
    by = _g(r, "synergy", "following_by_author")
    if isinstance(by, dict):
        for k in ("synergy", "planner"):
            if _num(by.get(k)) is not None:
                return _num(by.get(k))
    return None


_s("syn.following", _synergy_following, "The body follows commands written through the synergy map",
   (-1, "the body turns against the command"), (1, "exactly as commanded"), FOLLOW_DISPLAY,
   floor=(0, "the turn is unrelated to the command"), goal=FOLLOW_GOAL, refs=FOLLOW_REFS,
   goal_why=FOLLOW_GOAL_WHY, block="synergy", flag="motor_synergy",
   note="direction match over each synergy-written command's 250 ms life, read from the record's split by "
        "who wrote the command; not yet measured as a count",
   field="synergy.following_by_author")
_s("syn.following_all", _field("synergy", "following_mean"),
   "The body follows every graded command, whoever wrote it (a stand-in for item 5)",
   (-1, "the body turns against the command"), (1, "exactly as commanded"), FOLLOW_DISPLAY,
   floor=(0, "the turn is unrelated to the command"), goal=FOLLOW_GOAL, refs=FOLLOW_REFS,
   goal_why=FOLLOW_GOAL_WHY, block="synergy", flag="motor_synergy",
   note="babble and reflex commands included, so it is not the synergy's own following; not yet measured as "
        "a count", field="synergy.following_mean")
for _author in ("planner", "reflex", "reflex_noise", "external_rest", "acting_snapshot", "unknown"):
    _author_label = {'external_rest':'scheduled hold/rest', 'reflex_noise':'motor controller with exploration',
                     'reflex':'motor controller without exploration', 'acting_snapshot':'saved acting model',
                     'unknown':'unidentified controller'}.get(_author, _author)
    _s("syn.executed." + _author,
       _field("synergy", "following_by_author", _author, "above_chance_per_100"),
       _author_label.capitalize() + ": body-part commands ending closer above the random-axis comparison",
       (-100, "100 fewer than chance"), (100, "100 more than chance"),
       D_unit("part-commands above chance per 100"), floor=(0, "random-axis geometric control"),
       block="synergy", flag="motor_synergy",
       field="synergy.following_by_author." + _author + ".above_chance_per_100")
_s("syn.explained", _field("synergy", "explained_share"),
   "Posture changes the synergy map reproduces before learning from them",
   (0, "no better than asking for no change"), (1, "the whole change"),
   D_between(0, 1, "no better than asking for no change", "the whole change"), block="synergy",
   flag="motor_synergy", note="can be below 0; a map reproducing the average change scores high while "
                              "controlling nothing", field="synergy.explained_share")
_s("syn.dim", _field("synergy", "effective_dim"), "Diversity of the learned movement codes",
   (1, "variation along one internal direction"), None, D_count("directions"), block="synergy", flag="motor_synergy",
   note="estimated independent directions of variation in the compressed movement representation; higher means more diversity, not more skill. The upper end is the configured size of that representation", field="synergy.effective_dim")
_s("vest.gain", _field("vestibular", "improvement_over_persistence"),
   "Vestibular: next model step predicted better than \"nothing changes\"",
   (0, "no better than predicting that nothing changes"), (1, "exact"),
   D_between(0, 1, "no better than 'nothing changes'", "exact"),
   floor=(0, "predicting that nothing changes"), block="vestibular", flag="vestibular_sense",
   field="vestibular.improvement_over_persistence")
_s("vest.trunk_roll", lambda r: (lambda v: None if v is None else abs(v))(_num(_g(r, "vestibular", "trunk_roll_deg"))),
   "Trunk roll from lying on the back, from the vestibular input", (0, "on its back"), (180, "belly down"),
   D_unit("degrees", short="°"), block="vestibular", flag="vestibular_sense", field="vestibular.trunk_roll_deg")
_s("vest.support_trunk", _field("vestibular", "support_trunk"), "Weight carried by the trunk",
   (0, "none"), (1, "all of it"), D_per100("parts of its weight"), block="vestibular",
   flag="vestibular_sense", field="vestibular.support_trunk")
_s("machine.commit", _field("machine", "commit_gb"), "Memory committed on the machine", (0, "none"), None,
   D_unit("GiB"), limit_series="machine.commit_limit", block="machine", field="machine.commit_gb")
_s("machine.commit_limit", _field("machine", "commit_limit_gb"), "The machine's commit limit", (0, "none"),
   None, D_unit("GiB"), block="machine", field="machine.commit_limit_gb")
_s("machine.gpu", _field("machine", "gpu_used_mib"), "Card memory in use (all processes)", (0, "none"), None,
   D_unit("GiB", mul=1 / 1024.0), limit_series="machine.gpu_total", block="machine", field="machine.gpu_used_mib")
_s("machine.gpu_total", _field("machine", "gpu_total_mib"), "The card's memory", (0, "none"), None,
   D_unit("GiB", mul=1 / 1024.0), block="machine", field="machine.gpu_total_mib")
_s("machine.ram", _field("machine", "ram_available_gb"), "Memory still available", (0, "none"), None,
   D_unit("GiB"), block="machine", field="machine.ram_available_gb")
_s("machine.step_ms", _field("machine", "step_ms"), "Time per model step", (0, "instant"), None,
   D_unit("ms"), better="lower", block="machine", field="machine.step_ms")
_s("liveness.below", _below_line, "Layers with under half their units responding",
   (0, "every reported layer responds"), None, D_count("layers"), better="lower", block="liveness",
   note="every liveness report in the record (DESIGN.md section 10), against the 50-in-100 line",
   field="liveness.layers and every block's liveness")
_s("liveness.min", _min_live, "The least-responsive reported layer", (0, "no unit responds"),
   (1, "every unit responds"), D_per100("units"), floor=(LIVE_LINE, "the section 10 line"),
   block="liveness", field="liveness.layers and every block's liveness")
# the parts the lead's sweep found dead or locked on ckpt_step_344400, read from the model's
# permanent liveness block once the repair lane writes it (names as the sweep reported them)
KNOWN_DEAD = ("tenure_heads.pose.0", "wellness.approximate.net.0", "motor_controller.residual_blocks.0.0",
              "recon_decoders.vision.0", "recon_decoders.tactile.0", "persistent_memory.outcome.0")
# NO TITLE IS A BARE CODE PATH (the reading gate): each chart names the part in the words the Model
# health card uses (development_dashboard._part, held equal by test_liveness_part_names.py), keeps
# the code path beside it, and says what the part does, read from the model code:
# mimo_bridge/lem_model.py (tenure_heads["pose"], recon_decoders), lem/model/wellness.py (the
# approximate evaluator), lem/model/motor_controller.py, lem/model/persistent_memory.py (outcome,
# predict_needs) and mimo_bridge/memory_integration.py (measured_needs)
KNOWN_DEAD_WORDS = {
    "tenure_heads.pose.0": ("the tenure pose head", "from the model step a command began and the command itself, "
                            "it predicts how far each body part moves over the command's whole tenure (250 ms)"),
    "wellness.approximate.net.0": ("the wellness predictor, first layer",
                                   "the learned copy of wellness; the planner scores imagined model steps with it, "
                                   "since they have no skin or joint reading of their own"),
    "motor_controller.residual_blocks.0.0": ("motor controller, block 1",
                                             "the first residual block of the motor controller, the network "
                                             "that proposes a command for each body part"),
    "recon_decoders.vision.0": ("the vision reconstruction decoder", "it rebuilds the present vision token "
                                "from the forward model's output, a check that the output still contains "
                                "the present model step"),
    "recon_decoders.tactile.0": ("the tactile reconstruction decoder", "it rebuilds the present tactile token "
                                 "from the forward model's output, a check that the output still contains "
                                 "the present model step"),
    "persistent_memory.outcome.0": ("the persistent memory's outcome head", "from the memory's state and the "
                                    "command, it predicts five measured needs of the body: pain, pleasant touch, "
                                    "physical capacity, skin pressure and skin shear (sideways drag)"),
}
for _part in KNOWN_DEAD:
    _name, _role = KNOWN_DEAD_WORDS[_part]
    _s(f"live.{_part}", _field("liveness", "layers", _part), f"Units responding: {_name} ({_part}): {_role}",
       (0, "no unit responds"), (1, "every unit responds"), D_per100("units"),
       floor=(LIVE_LINE, "the section 10 line"), block="liveness", field=f"liveness.layers.{_part}")


def _block_min(block, key):
    """The smallest reading of a block's `liveness` or `pool_spread` sub-block (field contract
    2026-09-23): the module's least-responsive layer, or its most locked pool."""
    def get(r):
        sub = _g(r, block, key)
        if not isinstance(sub, dict):
            return None
        vals = [_num(v) for v in sub.values() if _num(v) is not None]
        return min(vals) if vals else None
    return get


# one lineage series per new module (reviews/dashboards.md M10: "Add a lineage series per module"):
# each module's least-responsive layer against the section 10 line, and its most locked pool
LIVE_MODULES = (("efference", "the efference heads", "efference_heads"),
                ("contact_control", "the contact head", "contact_head"),
                ("synergy", "the synergy map", "motor_synergy"),
                ("vestibular", "the vestibular encoder", "vestibular_sense"))
for _blk, _name, _flag in LIVE_MODULES:
    # each module names itself in its scale's ends, so the liveness card's panels never print the
    # same sentence twice (the card's own text defines a responding unit)
    _s(f"live.{_blk}", _block_min(_blk, "liveness"), f"Units responding in {_name}'s least-responsive layer",
       (0, f"no unit of that layer of {_name} responds"), (1, "every unit of it responds"),
       D_per100("units"), floor=(LIVE_LINE, "the section 10 line"), block=_blk, flag=_flag,
       field=f"{_blk}.liveness")
    _s(f"pool.{_blk}", _block_min(_blk, "pool_spread"), f"Attention spread of {_name}'s most locked pool",
       (0, "it reads one input"), (1, "every input evenly"),
       D_between(0, 1, "it reads one input", "every input evenly"), block=_blk, flag=_flag,
       note="DESIGN.md section 10 names no line for attention spread", field=f"{_blk}.pool_spread")


def signature():
    """Changes whenever the registry changes, so cached rows are re-extracted."""
    text = json.dumps([CACHE_VERSION, list(SERIES)])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def extract(record):
    out = []
    for spec in SERIES.values():
        try:
            value = spec["get"](record)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            value = None
        out.append(None if value is None else float(f"{value:.6g}"))
    return out


# ------------------------------------------------------------------ the chart cards
# One card per addendum item (section 19, items 1-10), the parts that stopped responding, and
# the machine. Items 7-9 and part of 10 are probe results (DESIGN.md section 9); their panels come
# from the probe index, not from these series.
CARDS = [
    dict(id="item1", item=1, title="1. Does it choose the action it predicts will feel best?",
         what='Compares the chosen movement with the alternatives predicted to improve the body’s condition. The current planner aims toward remembered states, so this can reveal a mismatch in its objective.',
         panels=[dict(title="Knowing its own executed command improves the next-model-step prediction (against "
                            "another body's absolute command)",
                      keys=["act.vision", "act.tactile", "act.proprioception"]),
                 dict(title="It chose the plan it predicted would feel best", keys=["cc.chosen_best", "cc.tied"]),
                 dict(title="How far apart the candidates were predicted to feel",
                      keys=["cc.gap", "cc.spread", "cc.tie_band"])]),
    dict(id="item2", item=2, title="2. Does the body do what it expected its command to do?",
         what='Compares actual movement with predictions using the actual command or a different command from the same body state. Agreement includes direction and size; it is not a count of successful actions.',
         panels=[dict(title="What the body did matches what it expected", keys=["ctl.agreement"]),
                 dict(title="Commands scored", keys=["ctl.measured"]),
                 dict(title="The body turns the way it was commanded (every command)", keys=["follow"]),
                 dict(title="Where each body part goes: predicted before the command ran", keys=["eff.body"])]),
    dict(id="item3", item=3, title="3. Are its prediction errors falling?",
         what='Tracks prediction errors. A historical experiment with learning disabled also changed as its experiences changed. Its range is a caution, not a matched test proving learning now.',
         panels=[dict(title="Recent change in prediction error", keys=["lp.value"]),
                 dict(title="Next model step predicted better than \"nothing changes\"",
                      keys=["pers.vision", "pers.tactile", "pers.proprioception"])]),
    dict(id="item4", item=4, title="4. Is the goal picked by one wellness channel only because its numbers are larger?",
         what='Shows which internal score contributes most to choosing a remembered body state as a goal. A large contribution does not establish that the resulting movements are useful.',
         panels=[dict(title="Strongest score: actual contribution versus assigned importance",
                      keys=["wr.lead", "wr.lead_weight"]),
                 dict(title="Goals chosen from the oldest and newest remembered states",
                      keys=["wr.age_oldest", "wr.age_newest"])]),
    dict(id="item5", item=5, title="5. Does the body follow the commands the synergy map writes?",
         what='Tests whether using motor synergy improves movement. Motor synergy is the network learning which body parts move together. Training it and allowing it to issue commands are separate settings.',
         panels=[dict(title="Executed commands ending closer, above their random-axis control, by author",
                      keys=["syn.executed." + a for a in ("planner", "reflex", "reflex_noise")]),
                 dict(title="Other executed authors, above their random-axis control",
                      keys=["syn.executed." + a for a in ("external_rest", "acting_snapshot", "unknown")]),
                 dict(title="Legacy direction-match readings (a different measure)",
                      keys=["syn.following", "syn.following_all"]),
                 dict(title="Posture changes the map reproduces (not the item's measurement)", keys=["syn.explained"]),
                 dict(title="Independent directions the map uses", keys=["syn.dim"])]),
    dict(id="item6", item=6, title="6. Is a touch of its own body foreseen before it happens?",
         what='Tests whether the model predicts a new self-touch before it happens, compared with guessing from past touch frequency and using a different movement command.',
         panels=[dict(title="Own-body touch foreseen before it begins", keys=["ccx.onset_self"]),
                 dict(title="New self-touches predicted with its own command", keys=["ccx.paired"]),
                 dict(title="Tactile prediction for bodies touching themselves", keys=["self_touch_act"])]),
    dict(id="item7", item=7, title="7. Does changing only the eye command shift the predicted view?", probe_item=7,
         what='Changes the eye command from the same starting state and measures whether the predicted view changes correctly. Predicting eye angle alone is a separate, easier test.',
         panels=[dict(title="Eye angles alone: where the eyes point 150 ms later", keys=["eff.gaze.3"])]),
    dict(id="item8", item=8, title="8. Do different commands give different, sensible predicted outcomes?", probe_item=8,
         what='Changes a limb command from the same starting state and compares predicted and actual outcomes. This tests prediction, not whether the model can choose a useful movement.',
         panels=[]),
    dict(id="item9", item=9, title="9. Does the predicted arm movement match what it sees and feels?",
         probe_item=9, what='Tests whether an arm movement entering view or touching the body changes the predicted vision and tactile tokens correctly.',
         panels=[]),
    dict(id="item10", item=10, title="10. Does it predict its own posture and balance?", probe_item=10,
         what='Tests predictions about posture and balance. Older tests use proprioception signals; newer tests read the vestibular input directly. Each test description identifies which was measured.',
         panels=[dict(title="Vestibular predicted better than \"nothing changes\"", keys=["vest.gain"]),
                 dict(title="How the bodies lie (0 degrees = on the back, 90 = on the side, 180 = belly down)",
                      keys=["posture.mean", "posture.max"])]),
    dict(id="liveness", item=None, title="Model health",
         what="Every liveness report in the record (DESIGN.md section 10): layers with under half their units "
              "responding on the window's samples, and each new module's least-responsive layer. Absent until "
              "the model reports it. What responding means: " + RESPONDS + ".",
         panels=[dict(title="Layers below the response warning line", keys=["liveness.below"]),
                 dict(title="The least-responsive reported layer", keys=["liveness.min"]),
                 dict(title="The efference heads' least-responsive layer", keys=["live.efference"]),
                 dict(title="The contact head's least-responsive layer", keys=["live.contact_control"]),
                 dict(title="The synergy map's least-responsive layer", keys=["live.synergy"])]),
    dict(id="machine", item=None, title="Training machine: speed and memory",
         what="Segment 31 was stopped when its speed fell under one third of its own first-hour speed; card "
              "memory spilling into system memory, or commit charge near its limit, is the suspected cause.",
         panels=[dict(title="Model steps per second", keys=["sps"]),
                 dict(title="Card memory in use, against the card's total", keys=["machine.gpu"]),
                 dict(title="Commit charge, against the machine's limit", keys=["machine.commit"]),
                 dict(title="Time per model step", keys=["machine.step_ms"])]),
]


def chart_keys():
    keys = []
    for card in CARDS:
        for panel in card["panels"]:
            for key in panel["keys"]:
                spec = SERIES[key]
                for k in (key, spec.get("floor_series"), spec.get("limit_series")):
                    if k and k not in keys:
                        keys.append(k)
    return keys


def other_keys():
    """Every series no card shows: the 'More gauges' section of Over time."""
    shown_keys = set(chart_keys())
    out = []
    for key, spec in SERIES.items():
        if key in shown_keys:
            continue
        if any(key == s.get("floor_series") or key == s.get("limit_series") for s in SERIES.values()):
            continue
        out.append(key)
    return out


# ------------------------------------------------------------------ the incremental reader
def cache_tag(sig):
    """The code-version tag a cache file carries in its name and its contents: the caller's
    signature (the series keys, or another page's extractor and its own source tag) and this file's
    SOURCE_TAG (the reader, and for the default extractor every series definition)."""
    return hashlib.sha1(f"{sig}|{SOURCE_TAG}".encode("utf-8")).hexdigest()[:16]


def _cache_file(cache_dir, path, tag):
    digest = hashlib.sha1(str(Path(path).resolve()).lower().encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / f"f_{digest}_{tag}.json"


def _prune_other_tags(cache):
    """Delete the cache files of the same stats file written under other tags, all but the newest
    one: the running server's older code and the newest code each keep their own file (the two
    versions of the open issue), and a third version's file does not pile up (every code version
    would otherwise leave one file per segment behind: 26 files, 2.5 MB, on H84 on 2026-09-23). The
    untagged file of the code before tags counts as another tag. Deleting one costs its version at
    most one re-read, never a wrong answer; a file another process holds open stays for the next pass."""
    digest = cache.name.split("_")[1].split(".")[0]
    others = []
    for p in cache.parent.glob(f"f_{digest}*.json"):
        if p != cache:
            try:
                others.append((p.stat().st_mtime, p))
            except OSError:
                continue
    for _mtime, old in sorted(others, reverse=True)[1:]:
        try:
            old.unlink()
        except OSError:
            pass


HEAD_BYTES = 4096


def _head(path, n=HEAD_BYTES):
    """Hash of the file's first n bytes: the same n must be used to compare, or a small file
    that grew would always look rewritten."""
    with open(path, "rb") as stream:
        return hashlib.sha1(stream.read(n)).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.new")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


_MEMO: dict = {}


def read_rows(path, cache_dir, extractor=extract, sig=None):
    """(rows, meta, stats) for one stats file: rows = [step, timestamp, [values]] per complete
    `mimo` record in file order; meta = launch records. Reads only bytes not read before.
    The newest result per file is also kept in memory, so an unchanged file costs one stat().
    The cache file is named and stamped with cache_tag(sig); a cache with another tag is not read."""
    sig = sig or signature()
    path = Path(path)
    stats = dict(bytes_read=0, from_cache=False, restarted=False)
    try:
        st = path.stat()
    except OSError:
        return [], {}, dict(stats, missing=True)
    tag = cache_tag(sig)
    memo_key = (str(path.resolve()).lower(), str(Path(cache_dir).resolve()).lower(), sig, tag)
    memo = _MEMO.get(memo_key)
    if memo and memo["size"] == st.st_size and memo["mtime_ns"] == st.st_mtime_ns:
        return memo["rows"], memo["meta"], dict(stats, from_cache=True)
    record, stats = _read_rows_disk(path, st, cache_dir, extractor, sig, stats, memo)
    if record is None:
        return [], {}, stats
    _MEMO[memo_key] = record
    return record["rows"], record["meta"], stats


def _read_rows_disk(path, st, cache_dir, extractor, sig, stats, cached=None):
    """Read what is new since `cached` (the in-memory copy, else the cache file); returns the
    new cache record and the read statistics."""
    tag = cache_tag(sig)
    cache = _cache_file(cache_dir, path, tag)
    if cached is None:
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
    # the tag in the contents is checked as well as the name: a copied or renamed file from other
    # code is refused, never read as this code's rows
    usable = (isinstance(cached, dict) and cached.get("signature") == sig and cached.get("tag") == tag
              and cached.get("path") == str(path.resolve()))
    if usable and cached["size"] == st.st_size and cached["mtime_ns"] == st.st_mtime_ns:
        return cached, dict(stats, from_cache=True)
    start, rows, meta = 0, [], {}
    try:
        head_len = min(HEAD_BYTES, st.st_size)
        head = _head(path, head_len)
        same_start = usable and cached.get("head_len") is not None and \
            _head(path, cached["head_len"]) == cached.get("head")
    except OSError:
        return None, dict(stats, missing=True)
    if usable and 0 < cached["offset"] <= st.st_size and same_start:
        # copies: a caller may still hold the previous lists
        start, rows, meta = cached["offset"], list(cached["rows"]), json.loads(json.dumps(cached["meta"]))
    elif usable:
        stats["restarted"] = True
    position = start
    with path.open("rb") as stream:
        stream.seek(start)
        for line in stream:
            if not line.endswith(b"\n"):
                break          # a write in progress: read it next time
            position += len(line)
            prefix = line[:48]
            if b'"type": "mimo"' in prefix:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                step, stamp = _num(record.get("step")), _num(record.get("timestamp"))
                if step is None or stamp is None or record.get("type") != "mimo":
                    continue
                rows.append([int(step), round(stamp, 1), extractor(record)])
            elif b'"type": "launch"' in prefix:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                meta.setdefault("launches", []).append(
                    {k: record.get(k) for k in ("timestamp", "step", "code_commit", "resumed_from", "code_dirty")})
    stats["bytes_read"] = position - start
    record = dict(signature=sig, tag=tag, path=str(path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns,
                  offset=position, head=head, head_len=head_len, rows=rows, meta=meta)
    try:
        _atomic_json(cache, record)
        _prune_other_tags(cache)
    except OSError:
        pass                   # a full or read-only disk costs a re-read, never a wrong answer
    return record, stats


# ------------------------------------------------------------------ launch settings
_PATHLIKE = {"out", "resume", "checkpoint_storage", "caregiver_record_dir", "caregiver_speech_dir",
             "caregiver_replay", "self_choice_receipt", "tag", "seed"}
TRAINER = REPO / "Source/Python/mimo_bridge/train_mimo.py"
ENV_SOURCE = REPO / "Source/Python/mimo_bridge/env.py"
# settings a resumed world carries over from its saved episode (continuation_state.
# migrate_env_kwargs): from commit bdbfb7895 on, a resume applied these two; every other env
# setting changed at a resume is still not applied (it warns). From commit 04c84682f on (fix-ops,
# DESIGN D2 note) a plain resume applies NONE of them again: the saved worlds keep their settings
# unless the launch passes --resume-env-settings launch (args.json resume_env_settings).
# The later names became migratable in the commit that added their flag (hand_target_mode in
# ecfc9060f; parent_relative_targets, reach_projection and position_rows with the motor interface
# fixes of 2026-09-25), so no segment ran one of them without the migration; before, the page called a
# deployed hand_target_mode "ignored by the resumed worlds". Kept equal to
# continuation_state.ENV_KWARGS_MIGRATABLE by test_dashboard_page_text.
ENV_MIGRATED = ("toy_hand_off", "finger_armature", "hand_target_mode", "parent_relative_targets",
                "reach_projection", "position_rows")
ENV_FIX_COMMIT = "bdbfb7895"
ENV_SAVED_DEFAULT_COMMIT = "04c84682f"
_DEFAULTS_CACHE: dict = {}
# The skin fixes' two settings, migratable from the commits that added their flags
# (contact_footprint_guard in 784f0b6c3, rubbing_calibration in f256d4886; merged into production at
# f4f335580). Added apart from the tuple above so a branch that extends the tuple merges beside it.
ENV_MIGRATED += ("contact_footprint_guard", "rubbing_calibration")
# The touch layout's repeatability switch (touch_inside_rays.py), migratable from the commit that added its flag.
ENV_MIGRATED += ("touch_inside_rays",)
# The finger command's meaning (--finger-joint-servo, action_decoder.py), migratable from the commit that added its flag.
ENV_MIGRATED += ("finger_joint_servo",)


def _flag_text(value):
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def trainer_defaults(path=TRAINER):
    """{flag name: default} read from train_mimo.parse_args's add_argument calls without importing
    the trainer (a literal default, False for store_true, None otherwise); cached on the file's
    modification time."""
    import ast
    try:
        st = Path(path).stat()
    except OSError:
        return {}
    key = (str(path), st.st_mtime_ns)
    if key in _DEFAULTS_CACHE:
        return _DEFAULTS_CACHE[key]
    out = {}
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("--")]
        if not names:
            continue
        dest = names[0][2:].replace("-", "_")
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if "dest" in kw and isinstance(kw["dest"], ast.Constant):
            dest = kw["dest"].value
        action = kw.get("action")
        if isinstance(action, ast.Constant) and action.value == "store_true":
            out[dest] = False
        elif isinstance(action, ast.Constant) and action.value == "store_false":
            out[dest] = True
        elif "default" in kw:
            try:
                out[dest] = ast.literal_eval(kw["default"])
            except ValueError:
                out[dest] = "<computed>"
        else:
            out[dest] = None
    _DEFAULTS_CACHE.clear()
    _DEFAULTS_CACHE[key] = out
    return out


def env_setting_names(path=ENV_SOURCE):
    """BridgeEnv's constructor parameters: the launch settings a world is built from."""
    import ast
    key = ("env", str(path))
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        return set()
    cached = _DEFAULTS_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    names = set()
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "BridgeEnv":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        names = {a.arg for a in item.args.args + item.args.kwonlyargs} - {"self"}
    except (OSError, SyntaxError, ValueError):
        names = set()
    _DEFAULTS_CACHE[key] = (mtime, names)
    return names


def _effective(args, key, defaults):
    """A setting's value in force: a flag absent from a run's args.json ran at its default."""
    if key in args:
        return args.get(key)
    return defaults.get(key)


_ANCESTOR_CACHE: dict = {}


def commit_has(commit, fix=ENV_FIX_COMMIT, root=REPO):
    """True when `fix` is an ancestor of (or is) `commit`, None when git cannot tell."""
    if not commit:
        return None
    key = (str(commit), fix)
    if key in _ANCESTOR_CACHE:
        return _ANCESTOR_CACHE[key]
    import subprocess
    try:
        rc = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", fix, str(commit)],
                            capture_output=True, timeout=10).returncode
        val = True if rc == 0 else False if rc == 1 else None
    except (OSError, subprocess.SubprocessError):
        val = None
    _ANCESTOR_CACHE[key] = val
    return val


def args_difference(child, parent, defaults=None, commit=None, resumed=True):
    """Launch settings whose value IN FORCE differs between a segment and the one it resumed
    from. A flag absent from a run's args.json ran at the trainer's default (reviews/dashboards.md
    M8), so "0.0004 (was not set)" for a default value is no difference. An env setting changed at
    a resume is labelled with what the resumed worlds did with it."""
    defaults = trainer_defaults() if defaults is None else defaults
    env_names = env_setting_names()
    has_fix = commit_has(commit) if resumed else None
    # from 04c84682f on the migration applies only with --resume-env-settings launch
    saved_default = commit_has(commit, ENV_SAVED_DEFAULT_COMMIT) if resumed and has_fix else None
    applies = (has_fix if not saved_default else
               str(child.get("resume_env_settings") or "saved").lower() == "launch")
    if has_fix and saved_default is None:
        applies = None
    out = []
    for key in sorted(set(child) | set(parent)):
        if key in _PATHLIKE:
            continue
        before, after = _effective(parent, key, defaults), _effective(child, key, defaults)
        if before == after:
            continue
        row = dict(flag="--" + key.replace("_", "-"), was=_flag_text(before), now=_flag_text(after))
        default_note = ", the trainer's default: the flag was newer than that segment" if key not in parent else ""
        if resumed and key in env_names:
            if key in ENV_MIGRATED and applies:
                row["note"] = ("applied to the resumed worlds (--resume-env-settings launch)" if saved_default
                               else "applied to the resumed worlds (commit bdbfb7895 migrates it)")
            elif key in ENV_MIGRATED and applies is None:
                row["note"] = ("an env setting; whether the resumed worlds applied it is unknown (the "
                               "segment's code commit is not recorded)")
            elif key in ENV_MIGRATED and saved_default:
                row["note"] = ("launch setting, not applied: from commit 04c84682f a resume keeps the saved "
                               "worlds' settings unless it is launched with --resume-env-settings launch")
            else:
                row["note"] = ("launch setting, ignored by the resumed worlds: a resumed world keeps its "
                               "saved setting (measured 2026-09-22 on segment 31's 8 saved worlds)")
        row["text"] = (f"{row['flag']} {row['now']} (was {row['flag']} {row['was']}{default_note})"
                       + (f": {row['note']}" if row.get("note") else ""))
        out.append(row)
    return out


def code_text(commit, dirty=None):
    """The code a segment ran, with what the record cannot say."""
    if not commit:
        return "code commit not recorded"
    if dirty is True:
        return f"commit {commit} with uncommitted changes"
    if dirty is False:
        return f"commit {commit} (no uncommitted changes)"
    return f"commit {commit} (uncommitted changes not recorded)"


_REGISTRY_CACHE = {}


def registry_commits(root):
    """{run name: [code_commit, ...]} from weights_registry.jsonl, cached on its size and mtime."""
    path = Path(root) / "Saved/mimo_bridge/weights_registry.jsonl"
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[key]
    commits = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("run") and row.get("code_commit"):
                    commits.setdefault(row["run"], [])
                    if row["code_commit"] not in commits[row["run"]]:
                        commits[row["run"]].append(row["code_commit"])
    except OSError:
        return {}
    _REGISTRY_CACHE.clear()
    _REGISTRY_CACHE[key] = commits
    return commits


def short_label(run_name, root_name):
    if root_name and run_name.startswith(root_name) and run_name != root_name:
        return run_name[len(root_name):].lstrip("_") or run_name
    return run_name


# the flags that write a DESIGN.md section 3 block, with the value that turns each on
BLOCK_FLAGS = {"causal_chain": "on", "efference_heads": "on", "vestibular_sense": "on",
               "wellness_rank": "normalized", "controllability": "on", "learning_progress": "on",
               "contact_head": "on", "motor_synergy": "on", "contact_control_wellness": ">0"}


def flag_on(args, flag, on_value=None):
    """Whether a run's launch settings turn a flag on (None when the flag is absent)."""
    if flag not in (args or {}):
        return None
    value = args.get(flag)
    on_value = on_value or BLOCK_FLAGS.get(flag, "on")
    if on_value == ">0":
        return bool(_num(value) and _num(value) > 0)
    if isinstance(value, bool):
        return value
    return str(value).lower() == str(on_value).lower()


def segments(root, bounds, runs_rows=None):
    """Segment markers, oldest first: name, step range, code commit, the settings changed (their
    values in force), and which section 3 flags each segment ran with."""
    runs = list(reversed(list(bounds)))
    root_name = Path(runs[0]).name if runs else ""
    commits = registry_commits(root)
    defaults = trainer_defaults()
    out, parent_args = [], None
    for index, run in enumerate(runs):
        args = _read_json(Path(run) / "args.json", {}) or {}
        rows, meta = (runs_rows or {}).get(run, ([], {}))
        launches = [m for m in meta.get("launches", []) if bounds[run] is None or (_num(m.get("step")) or 0) <= bounds[run]]
        first = next((m for m in launches if m.get("code_commit")), None)
        commit = first.get("code_commit") if first else None
        dirty = first.get("code_dirty") if first else None
        if commit is None:
            commit = (commits.get(Path(run).name) or [None])[0]
        out.append(dict(index=index, run=Path(run).name, label=short_label(Path(run).name, root_name),
                        limit=bounds[run], first_step=rows[0][0] if rows else None,
                        last_step=rows[-1][0] if rows else None,
                        first_time=rows[0][1] if rows else None, last_time=rows[-1][1] if rows else None,
                        records=len(rows), code_commit=commit, code=code_text(commit, dirty),
                        restarts=max(0, len(launches) - 1),
                        changed=(args_difference(args, parent_args, defaults, commit)
                                 if parent_args is not None else []),
                        flags={f: flag_on(args, f) for f in BLOCK_FLAGS},
                        fresh=not args.get("resume")))
        parent_args = args
    return out


def seam_segment(segs, commit):
    """(the first segment, oldest first, whose recorded code commit contains `commit`, or None; how many
    segments before it had no code commit git could check)."""
    unknown = 0
    for seg in segs or []:
        has = commit_has(seg.get("code_commit"), commit)
        if has:
            return seg, unknown
        if has is None:
            unknown += 1
    return None, unknown


def seam_text(spec, segs):
    """(the sentence a series with a definition seam prints, the segment where the new definition begins
    or None); (None, None) for a series without a seam."""
    seam = spec.get("seam")
    if not seam:
        return None, None
    seg, unknown = seam_segment(segs, seam["commit"])
    text = f"This gauge changed at commit {seam['commit']} ({seam['date']}): {seam['change']} ({seam['source']})."
    if seg is not None and seg.get("first_step") is not None:
        text += (f" Readings from segment {seg['label']} (lineage step {seg['first_step']:,}) on are on the new gauge "
                 f"and the readings before it on the old one, so a step there is the change of gauge, not of the body.")
    elif seg is not None:
        text += (f" Segment {seg['label']} is the first to run it and has no records yet; every reading here is on the "
                 f"old gauge.")
    else:
        text += (" No segment of this lineage runs that commit yet, so every reading here is on the old gauge; the "
                 "first segment that runs it begins the new one.")
    if unknown:
        text += (f" {unknown} segment{'s' if unknown != 1 else ''} of this lineage{' before it' if seg else ''} "
                 f"recorded no code commit git could check, so {'they are' if unknown != 1 else 'it is'} counted "
                 f"on the old gauge without proof.")
    return text, seg


# ------------------------------------------------------------------ text written elsewhere
def printable(text):
    """A text written elsewhere (a probe's status, a producer's reason, a watcher's stop reason),
    shown only when it passes DESIGN.md section 11 and the readout scanner: plain_numbers.clean
    holds it to the bare-decimal rule, and unanchored_numbers adds a score of 1 or more with no unit,
    a multiplier with nothing named after it and a raw input value of a hidden unit (fix-scanner,
    commit 22b047f9b). None otherwise; the caller says where the text can be read."""
    from .check_no_bare_decimals import unanchored_numbers
    out = clean(text)
    return None if out is None or unanchored_numbers(out) else out


_ASIDE = re.compile(r"\s*\([^()]*\)")


def printable_parts(text):
    """(the parts of a text written elsewhere that pass `printable`, joined, or None when none
    does; how many parts were left out). A sentence that fails is first tried without each bracketed
    aside that fails on its own, so it keeps its point when only the aside prints numbers a reader
    cannot place; a sentence that still fails is left out whole. The body probe's own summary
    (2026-09-23 03:50) ends "none of its 256 hidden units responded (a unit responds when its input
    rises above -3; the highest input any unit received was -7)": the aside is left out, the point
    stays (fix-readout-dashboards)."""
    from .check_no_bare_decimals import SENTENCE_END
    kept, left_out = [], 0
    for sentence in SENTENCE_END.split(str(text or "").strip()):
        if not sentence.strip():
            continue
        ok = printable(sentence)
        if ok is None:
            asides = [m.group(0) for m in _ASIDE.finditer(sentence) if printable(m.group(0)) is None]
            trimmed = sentence
            for aside in asides:
                trimmed = trimmed.replace(aside, "", 1)
            ok = printable(trimmed) if asides and trimmed.strip() else None
            left_out += len(asides) if ok is not None else 1
        if ok is not None:
            kept.append(ok)
    return (" ".join(kept) or None), left_out


_TEXT_KEYS_KEPT = frozenset(("metric", "path", "checkpoint", "time"))


def _printable_tree(value, key=None):
    """A copy of a probe headline with every text a person could read passed through
    printable_parts (identifiers and paths kept as they are): the /api/lineage payload carries no
    text the page would not print."""
    if isinstance(value, dict):
        return {k: (v if k in _TEXT_KEYS_KEPT else _printable_tree(v, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_printable_tree(v, key) for v in value]
    if isinstance(value, str) and any(ch.isdigit() for ch in value):
        return printable_parts(value)[0]
    return value


def _left_out_note(n, path):
    """What a reader is told about the parts of a probe's own summary left out here."""
    if not n:
        return ""
    return (f" ({n} part{'s' if n != 1 else ''} left out here: {'it prints' if n == 1 else 'they print'} numbers a "
            f"reader cannot place; the whole summary is in {path or 'the probe result file'})")


# ------------------------------------------------------------------ probes (DESIGN.md section 9)
def latest_index(rows):
    """The newest index line per (probe, checkpoint): a re-run replaces the run before it
    (reviews/probes.md: the gaze probe was re-run on ckpt_step_344400 for a corrected sentence and
    both lines were plotted). Newest = latest `time`, then the later line in the file."""
    best = {}
    for order, row in enumerate(rows):
        key = (row.get("probe"), str(row.get("checkpoint") or row.get("path") or order))
        stamp = str(row.get("time") or "")
        prev = best.get(key)
        if prev is None or (stamp, order) >= (prev[0], prev[1]):
            best[key] = (stamp, order, row)
    return [v[2] for v in sorted(best.values(), key=lambda v: v[1])]


def probe_results(root, bounds):
    """{item: [probe series]} from Saved/mimo_bridge/gof_probes/index.jsonl, ancestry only, the
    newest line per (probe, checkpoint)."""
    from .development_dashboard import lineage_relation
    path = Path(root) / "Saved/mimo_bridge/gof_probes/index.jsonl"
    by_name = {Path(run).name: limit for run, limit in bounds.items()}
    out, other = {}, 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return dict(items={}, index_found=False, other_lineage=0, replaced=0)
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("headline"), dict):
            rows.append(row)
    # a line without a value is a run that produced no reading (the suite's failure record, value
    # None and status "failed: ..."): it never replaces a reading, and it is reported beside the
    # item when it is newer than the item's reading (verify-fix-dashboards 2026-09-23)
    valued = [r for r in rows if _num(r["headline"].get("value")) is not None]
    empty = [r for r in rows if _num(r["headline"].get("value")) is None]
    kept, kept_empty = latest_index(valued), latest_index(empty)

    def on_chain_step(row):
        step = _num(row.get("lineage_step"))
        on_chain = False
        if row.get("run") in by_name and step is not None:
            limit = by_name[row["run"]]
            on_chain = limit is None or step <= limit
        if not on_chain and row.get("checkpoint"):
            on_chain = lineage_relation(root, row["checkpoint"], bounds) == "ancestor_checkpoint"
        return on_chain, step
    for row in kept:
        on_chain, step = on_chain_step(row)
        if not on_chain:
            other += 1
            continue
        head = row["headline"]
        series = out.setdefault(str(row.get("item")), {})
        entry = series.setdefault(row.get("probe") or "probe", dict(
            probe=row.get("probe"), metric=head.get("metric"), unit=head.get("unit"), points=[]))
        entry["points"].append([int(step) if step is not None else None, _num(head.get("value"))])
        where = dict(lineage_step=step, checkpoint=row.get("checkpoint"), time=row.get("time"),
                     path=row.get("path"), status=row.get("status"))
        # the page's sentences are built from the probe's own headline (probe_words below); the
        # payload's copy keeps only the text the page would print (fix-readout-dashboards 2026-09-23)
        entry["_head"] = dict(head, **where)
        entry["latest"] = dict(_printable_tree(head), **where)
    for row in kept_empty:
        on_chain, step = on_chain_step(row)
        if not on_chain:
            other += 1
            continue
        series = out.setdefault(str(row.get("item")), {})
        entry = series.setdefault(row.get("probe") or "probe", dict(
            probe=row.get("probe"), metric=None, unit=None, points=[], latest=None))
        last = entry.get("latest")
        when = ((step if step is not None else -1), str(row.get("time") or ""))
        if last and ((_num(last.get("lineage_step")) if last.get("lineage_step") is not None else -1),
                     str(last.get("time") or "")) >= when:
            continue           # an older attempt the reading has since superseded
        prev = entry.get("failed")
        if prev is None or ((_num(prev.get("lineage_step")) if prev.get("lineage_step") is not None else -1),
                            str(prev.get("time") or "")) < when:
            entry["failed"] = dict(lineage_step=step, checkpoint=row.get("checkpoint"), time=row.get("time"),
                                   path=row.get("path"), status=str(row.get("status") or ""))
    items = {}
    for key, series in out.items():
        rows_ = list(series.values())
        for entry in rows_:
            entry.setdefault("latest", None)
            head = entry.pop("_head", None)
            entry["points"].sort(key=lambda p: (p[0] is None, p[0] or 0))
            entry["words"] = probe_words(head or entry["latest"], entry.get("probe")) if entry["latest"] else None
            if entry.get("failed"):
                entry["failed"]["text"] = failed_probe_text(entry)
                entry["failed"]["status"] = printable(entry["failed"]["status"])
                if entry["words"] is None:
                    label = PROBE_LABELS.get(str(entry.get("probe")), str(entry.get("probe") or "probe"))
                    entry["words"] = dict(label=label, big="unmeasured", sentence=entry["failed"]["text"], scale=None,
                                          goal="Distance to goal unmeasured.", stand_in=False, own=None,
                                          own_withheld=False, floor=None, share=None)
                else:
                    entry["words"]["failed"] = entry["failed"]["text"]
        items[key] = rows_
    return dict(items=items, index_found=True, other_lineage=other,
                replaced=(len(valued) - len(kept)) + (len(empty) - len(kept_empty)))


def failed_probe_text(entry):
    """One sentence for a probe run that produced no reading: which checkpoint, when, the status the
    suite wrote (shown only when it passes DESIGN.md section 11), and which reading stands instead."""
    f = entry.get("failed") or {}
    label = PROBE_LABELS.get(str(entry.get("probe")), str(entry.get("probe") or "probe"))
    step = f.get("lineage_step")
    where = (f"on lineage step {int(step):,}" if step is not None else "on an unrecorded checkpoint") + (
        f", {f['time']}" if f.get("time") else "")
    raw = str(f.get("status") or "")
    status = printable(raw)
    verb = "stopped with an error" if raw.lower().startswith("failed") else "produced no reading"
    why = (f" ({status.rstrip('.')})" if status else
           " (its status is in " + str(f.get("path") or "the probe index") + ")")
    last = entry.get("latest")
    if last:
        ls_ = last.get("lineage_step")
        return (f"The newest run of this probe ({where}) {verb}{why}; the reading above is from "
                + (f"lineage step {int(ls_):,}." if ls_ is not None else "an earlier run."))
    return f"The {label[:1].lower() + label[1:]}'s newest run ({where}) {verb}{why}, so this item has no reading."


# What each known probe headline measures, in the words the page uses (the probes' own unit strings
# say "token" and "handle", code names the page never defines: reviews/dashboards.md M7).
PROBE_METRICS = {
    "gaze_captured_share_vision_150ms": dict(
        label="Gaze probe",
        what="When only the eye command changed, the picture the model predicted for 150 ms later moved {pct} "
             "of the way the real picture moved",
        ends=("it ignores where the eyes are sent", "it foresees the whole change")),
    "body_captured_share_tenure_cm": dict(
        label="Body-control probe",
        what="When one limb was commanded to turn 20 degrees, the model's 250 ms prediction of where each body "
             "part would be foresaw {pct} of the real difference the command made",
        ends=("it ignores the command", "it foresees the whole change")),
    "binding_vision_share_hand_entering_view": dict(
        label="Arm-into-view probe",
        what="When the hand was sent into view, the model's next-model-step prediction of what the eyes see "
             "foresaw {pct} of the real change as the hand entered",
        ends=("its prediction does not move with the hand", "it foresees the whole change")),
    # the counterfactual form the binding probe writes since fix-probes (2026-09-23 02:56 re-run):
    # continuing the order against stopping the hand, from the same moment
    "binding_vision_counterfactual_hand_entering_view": dict(
        label="Arm-into-view probe",
        what="When the hand was sent into view, the model's next-model-step prediction of what the eyes see, for "
             "continuing the command against stopping the hand, foresaw {pct} of the real difference between "
             "continuing and stopping as the hand entered",
        ends=("the prediction does not depend on the command", "it foresees the whole difference")),
    # the body probe's headline once the build has the efference body head (gof_probe_body.py)
    "body_captured_share_efference_cm": dict(
        label="Body-control probe",
        what="When one limb was commanded to turn 20 degrees, the efference body head's 250 ms prediction of where "
             "each body part would be foresaw {pct} of the real difference the command made",
        ends=("it ignores the command", "it foresees the whole change")),
    "posture_neck_share_50ms": dict(
        label="Posture probe",
        what="When only the head was commanded to turn or tilt 15 degrees, the model's next-model-step proprioception "
             "prediction foresaw {pct} of the real difference; a stand-in, because the head's own part cannot "
             "be separated inside the proprioception reading",
        ends=("it ignores the command", "it foresees the whole change"), stand_in=True),
}


PROBE_LABELS = {"gaze": "Gaze probe", "body": "Body-control probe", "binding": "Arm-into-view probe",
                "posture": "Posture probe"}


def probe_words(head, probe=None):
    """The page's sentences for one probe headline: the value as a percentage with significant
    digits, the scale's ends, the goal line (the top of a share scale is not a goal), the distance,
    and whether it is a stand-in reading. The probe's own summary keeps only its parts that pass
    section 11 and the readout scanner (printable_parts), after its scientific numbers are
    rewritten, and says how many parts were left out and where the whole summary is."""
    head = head or {}
    metric = str(head.get("metric") or "")
    known = PROBE_METRICS.get(metric)
    value = _num(head.get("value"))
    unit_text = str(head.get("unit") or "")
    share_scale = "share" in unit_text or known is not None
    lived = str(head.get("lived") or "")
    # Older and direct vestibular tests share this ID; use the recorded sense.
    indirect = any(word in (unit_text + ' ' + lived).lower() for word in ('standing in', 'stand-in', 'proprioception', 'proxy')) or head.get('proxy')
    # the unit text is the probe result's own (gof_probe_posture.py names a vestibular reading in its
    # older words), matched here so older result files still read right; never page text
    direct_balance = metric == 'posture_neck_share_50ms' and not indirect and ('the balance sense' in unit_text or 'vestibular' in unit_text)
    stand_in = bool(not direct_balance and ((known or {}).get('stand_in') or 'proxy' in lived.lower() or head.get('proxy')))
    if direct_balance:
        known = dict(known or {}, label='Vestibular prediction test', what='After a head command, the vestibular prediction foresaw {pct} of the real change compared with holding still', stand_in=False)
    # the probe's code name ("binding") is never the page's word for it (reviews/dashboards.md M7)
    label = (known or {}).get("label") or PROBE_LABELS.get(str(probe), str(probe or "The probe"))
    if value is None:
        big, sentence = "unmeasured", f"{label}: no value in its headline."
    elif share_scale:
        big = pct(value, 2) + " of the way"
        if known:
            sentence = known["what"].format(pct=pct(value, 2)) + "."
        else:
            sentence = f"{label}: {pct(value, 2)} of the way from 'ignores the command' to " \
                       f"'foresees the whole change'."
    else:
        big = fixed(value, 2) + (" " + unit_text if unit_text else "")
        sentence = f"{label}: {big}."
    ends = (known or {}).get("ends") or ("it ignores the command", "it foresees the whole change")
    # a negative share is a prediction that moved against the real change (the re-run gaze probe of
    # 2026-09-23 01:46 reads below 0 on ckpt_step_344400); the scale says so beside it
    scale = (f"0% = {ends[0]}; 100% = {ends[1]}"
             + ("; below 0% = the prediction moved the opposite way from the real change"
                if value is not None and value < 0 else "") + ".") if share_scale else None
    floor, goal = _num(head.get("floor")), _num(head.get("goal"))
    top = 1.0 if share_scale else None
    if goal is None or (top is not None and goal >= top):
        goal_text = ('This tests whether changing a command changes the prediction correctly. It does not test whether the model can choose and repeat a useful movement. No working-skill reference has been measured.')
    else:
        dist = _num(head.get("distance_to_goal"))
        goal_text = (f"Goal line {pct(goal, 3) if share_scale else fixed(goal, 3)}"
                     + (f"; distance to it {pct(dist, 3) if share_scale else fixed(dist, 3)}." if dist is not None
                        else "; distance from a working skill has not been measured."))
    own, left_out = printable_parts(lived) if lived else (None, 0)
    if own is not None:
        own += _left_out_note(left_out, head.get("path"))
    return dict(label=label, big=big, sentence=sentence, scale=scale,
                goal=goal_text, stand_in=stand_in, own=own,
                own_withheld=bool(lived) and own is None, floor=floor,
                share=value if share_scale else None)


# ------------------------------------------------------------------ assembly
def _bucketed(segs, keys, max_buckets):
    index = {k: i for i, k in enumerate(SERIES)}
    cols = [index[k] for k in keys]
    steps = [row[0] for seg in segs for row in seg["rows"]]
    if not steps:
        return dict(step=[], time=[], segment=[]), {k: None for k in keys}
    low, high = min(steps), max(steps)
    width = max(1.0, (high - low) / max(1, max_buckets))
    xs, ts, sg, starts, first_times = [], [], [], [], []
    sums = {k: [] for k in keys}
    counts = {k: [] for k in keys}
    last = None
    for seg in segs:
        for step, stamp, values in seg["rows"]:
            bucket = (seg["index"], min(max(1, max_buckets) - 1, int((step - low) // width)))
            if bucket != last:
                xs.append(step); ts.append(stamp); sg.append(seg["index"])
                starts.append(step); first_times.append(stamp)
                for k in keys:
                    sums[k].append(0.0); counts[k].append(0)
                last = bucket
            xs[-1], ts[-1] = step, stamp
            for k, c in zip(keys, cols):
                v = values[c] if c < len(values) else None
                if v is not None:
                    sums[k][-1] += v
                    counts[k][-1] += 1
    out = {}
    for k in keys:
        if not any(counts[k]):
            out[k] = None
            continue
        out[k] = [float(f"{s / n:.5g}") if n else None for s, n in zip(sums[k], counts[k])]
    return dict(step=xs, time=ts, segment=sg, start_step=starts, start_time=first_times,
                counts=counts, aggregation='mean of available report windows'), out


def _latest(segs, key):
    col = list(SERIES).index(key)
    for seg in reversed(segs):
        for step, stamp, values in reversed(seg["rows"]):
            v = values[col] if col < len(values) else None
            if v is not None:
                return dict(value=v, step=step, time=stamp, run=seg["run"])
    return None


def public_def(key):
    spec = SERIES[key]
    out = {k: v for k, v in spec.items() if k != "get"}
    return out


def scale_text(spec):
    """The scale's two ends in the display form, in one sentence."""
    d = spec["display"]
    if d["kind"] == "between":
        text = f"0% = {d['lo_name']}; 100% = {d['hi_name']}."
        if spec.get('key', '').startswith(('act.', 'pers.', 'cc.pers.', 'eff.', 'vest.')) or 'improvement_over_' in str(spec.get('field') or ''):
            text += ' Below 0% means more prediction error than the named comparison; -100% means twice its error.'
        elif spec.get('key') == 'syn.explained':
            text += ' Below 0% means more reconstruction error than assuming no change; -100% means twice that error.'
        return text
    if d["kind"] in ("per100", "auc"):
        lo, hi = spec["lo"], spec.get("hi")
        return (f"{fixed(100 * lo[0])} = {lo[1]}" + (f"; {fixed(100 * hi[0])} = {hi[1]}" if hi else "")
                + f" (out of every 100 {d['noun']}).")
    if d["kind"] == "change":
        return "0% = errors not changing; above 0 = falling (getting better); below 0 = rising."
    lo, hi = spec["lo"], spec.get("hi")
    return f"{say_ref(spec, lo[0])} = {lo[1]}" + (f"; {say_ref(spec, hi[0])} = {hi[1]}" if hi else "") + "."


def goal_text(spec, value=None, name=None, before_seam=False):
    """The goal line and the distance to it, or the words that it is unmeasured and why. `name`
    (a panel's title) says which gauge an unmeasured goal belongs to, so two panels of one card
    never print the same sentence. `before_seam`: the reading is from before the series' definition
    seam, while its goal line was measured on the gauge after it, so no distance is given (a distance
    between two gauges is not a distance; fix-rows-r2-lineage_series)."""
    goal = spec.get("goal")
    if not goal:
        why = spec.get("goal_why")
        name = name or (spec["title"].split(":")[0] if ":" in spec["title"] else spec["title"])
        return f'What this measurement cannot establish: {why}.' if why else ''
    g = goal[0]
    text = f"Goal line {say_ref(spec, g)}: {goal[1]}."
    if value is None:
        return text
    if before_seam and spec.get("seam"):
        return text + (f" Distance not given: this reading is on the gauge before commit {spec['seam']['commit']}, "
                       f"and the goal line was measured on the gauge after it.")
    d = spec["display"]
    gap = abs(shown(d, g) - shown(d, value))
    higher = spec.get("better") != "lower"
    short = value < g if higher else value > g
    if d["kind"] == "between":
        gap_text = fixed(gap, 2) + " percentage points"
    elif d["kind"] in ("per100", "auc"):
        gap_text = per(gap / 100.0, d["noun"])
    else:
        gap_text = say(spec, abs(g - value))
    return text + f" Distance: {gap_text} {'short of' if short else 'past'} it."


def panel_texts(panel, defs, segs):
    """The sentences a chart panel prints: the latest reading of each line, the scale's ends, the
    goal line, and - when no segment carries the series - why, read from each segment's settings."""
    keys = [k for k in panel["keys"] if k in defs]
    if not keys:
        return dict(unmeasured="Unmeasured: this server does not define the series.")
    d0 = SERIES[keys[0]]
    latest = [(k, defs[k].get("latest")) for k in keys]
    have = [(k, l) for k, l in latest if l]
    if have:
        d0 = SERIES[have[0][0]]
    out = dict(scale=scale_text(d0) + (" " + d0["note"][0].upper() + d0["note"][1:] + "." if d0.get("note") else ""))
    if 'syn.following_all' in keys:
        out['scale'] = ('0% = turn unrelated to the command; 100% = exactly the commanded direction; negative = opposite direction. '
                        'The every-command line includes the current controller and exploration. Only the separately labeled synergy-command line can describe synergy control. Movement size is not scored.')
    if not have:
        flag = d0.get("flag")
        if flag:
            on = [s["label"] for s in segs if (s.get("flags") or {}).get(flag)]
            value = BLOCK_FLAGS.get(flag, "on")
            state = ("set to " + value) if value not in ("on", ">0") else ("above 0" if value == ">0" else "on")
            if on:
                out["unmeasured"] = (f"Unmeasured: segment{'s' if len(on) > 1 else ''} {', '.join(on)} ran with "
                                     f"--{flag.replace('_', '-')} {state}, but no record carries the "
                                     f"'{d0['block']}' block's {d0.get('field') or 'field'}.")
            else:
                out["unmeasured"] = (f"Unmeasured on every segment of this lineage: it is written only while "
                                     f"--{flag.replace('_', '-')} is {state}, and no segment's launch settings "
                                     f"(args.json) turn it {'on' if state == 'on' else 'to that value'}.")
        elif d0.get("field"):
            out["unmeasured"] = (f"Unmeasured: no record of this lineage carries {d0['field']}"
                                 + (f" ({d0['note']})" if d0.get("note") and "only" in d0["note"] else "") + ".")
        else:
            out["unmeasured"] = "Unmeasured: no segment of this lineage carries this field."
        out['missing_summary'] = ('The feature is enabled, but these training records contain no usable measurement.'
                                  if flag and segs and (segs[-1].get('flags') or {}).get(flag) else
                                  'No measurement is recorded in this model’s training history.')
        out["goal"] = goal_text(d0, name=panel.get("title"))
        return out
    seg = next((s for s in segs if s["run"] == have[0][1]["run"]), None)
    parts = []
    for k, l in have:
        spec = SERIES[k]
        name = spec["title"].split(":")[0] if ":" in spec["title"] else spec["title"]
        text = say(spec, l["value"], full=len(keys) == 1)
        parts.append((name + ": " if len(keys) > 1 else "") + text)
    for k in keys:
        spec = SERIES[k]
        fs = spec.get("floor_series")
        if fs and defs.get(fs, {}).get("latest"):
            parts.append(f"{SERIES[fs]['title'].lower()}: {say(SERIES[fs], defs[fs]['latest']['value'], full=False)}")
        ls_ = spec.get("limit_series")
        if ls_ and defs.get(ls_, {}).get("latest"):
            parts.append(f"against {SERIES[ls_]['title'].lower()} of {say(SERIES[ls_], defs[ls_]['latest']['value'])}")
    out["latest"] = (f"Latest, lineage step {have[0][1]['step']:,}"
                     + (f" (segment {seg['label']})" if seg else "") + ": " + "; ".join(parts) + ".")
    seam, seam_seg = seam_text(d0, segs)
    # the latest reading is on the old gauge when no segment runs the change or its segment is earlier
    before = bool(seam) and (seam_seg is None or seg is None or seg["index"] < seam_seg["index"])
    out["goal"] = goal_text(d0, have[0][1]["value"] if len(keys) == 1 and have[0][0] == keys[0] else None,
                            name=panel.get("title"), before_seam=before)
    if d0.get("band"):
        b = d0["band"]
        out["band"] = f"Shaded: {b[2]} read {say_ref(d0, b[0])} to {say_ref(d0, b[1])}; a reading inside it " \
                      f"is not evidence of learning."
    if seam:
        out["seam"] = seam
    return out


def build(root, training=None, bounds=None, cache_dir=None, keys=None, max_buckets=MAX_BUCKETS, now=None):
    """The /api/lineage payload: aligned, downsampled series over the lineage plus segment
    markers, series definitions, chart cards (with their sentences) and probe results."""
    from .development_dashboard import training_state, ancestry_bounds
    root = Path(root)
    now = time.time() if now is None else now
    with _LOCK:
        started = time.perf_counter()
        training = training or training_state(root, now)
        bounds = ancestry_bounds(root, training) if bounds is None else bounds
        cache_dir = Path(cache_dir) if cache_dir else root / "Saved/mimo_bridge/lineage_series"
        sig = signature()
        runs_rows, read = {}, dict(files=0, bytes_read=0, from_cache=0, dropped_fork_records=0, missing=0)
        for run, limit in bounds.items():
            rows, meta, stats = read_rows(Path(run) / "replay_training_stats.jsonl", cache_dir, sig=sig)
            kept = [r for r in rows if limit is None or r[0] <= limit]
            read["files"] += 1
            read["bytes_read"] += stats.get("bytes_read", 0)
            read["from_cache"] += bool(stats.get("from_cache"))
            read["missing"] += bool(stats.get("missing"))
            read["dropped_fork_records"] += len(rows) - len(kept)
            runs_rows[run] = (kept, meta)
        segs = segments(root, bounds, runs_rows)
        chain = [dict(index=s["index"], run=s["run"], rows=runs_rows[run][0])
                 for s, run in zip(segs, reversed(list(bounds)))]
        all_keys = keys == "all"
        keys = [k for k in (list(SERIES) if all_keys else (keys or chart_keys())) if k in SERIES]
        x, values = _bucketed(chain, keys, max_buckets)
        defs = {}
        for k in keys:
            defs[k] = public_def(k)
            defs[k]["latest"] = _latest(chain, k)
            if defs[k].get("seam"):
                # where the chart marks the seam: the first segment whose code contains the change
                _text, seg = seam_text(defs[k], segs)
                defs[k]["seam_at"] = (dict(step=seg["first_step"], label=seg["label"])
                                      if seg is not None and seg.get("first_step") is not None else None)
        cards = []
        for card in CARDS:
            c = dict(card, panels=[dict(p, texts=panel_texts(p, defs, segs)) for p in card["panels"]])
            # one reason, said once per card: panels whose series share a flag gave the same
            # "unmeasured" sentence two or three times in one card on the real lineage (item 5 three
            # times; verify-fix-dashboards 2026-09-23, reading gate)
            said = set()
            for p in c["panels"]:
                plain = p['texts'].get('missing_summary')
                if plain and plain in said:
                    p['texts']['missing_summary'] = 'No measurement recorded.'
                elif plain:
                    said.add(plain)
                why = p["texts"].get("unmeasured")
                if why and why in said:
                    p["texts"]["unmeasured"] = "Unmeasured, as above."
                elif why:
                    said.add(why)
            cards.append(c)
        more = []
        if all_keys:
            for k in other_keys():
                p = dict(title=SERIES[k]["title"], keys=[k])
                more.append(dict(p, texts=panel_texts(p, defs, segs)))
        read["seconds"] = round(time.perf_counter() - started, 3)
        return dict(schema=1, generated_at=now, run=training.get("run"), run_name=training.get("run_name"),
                    x=x, series=values, defs=defs, segments=segs, cards=cards, more=more,
                    probes=probe_results(root, bounds), read=read)


# ------------------------------------------------------------------ the ten items, now
def _flag_state(args, flag, on_value=None):
    if flag not in args:
        return "absent from this run's launch settings (the flag is newer than the run)"
    on = flag_on(args, flag, on_value)
    value = _flag_text(args.get(flag))
    if on:
        return "on" + (f" ({value})" if value not in ("on", "True") else "")
    return "off" if value in ("off", "not set", "0", "0.0") else f"off (set to {value})"


def _line(kind, text):
    return dict(kind=kind, text=text)


def _say_field(key, record, full=True):
    spec = SERIES[key]
    try:
        value = spec["get"](record) if record else None
    except (TypeError, ValueError, ZeroDivisionError):
        value = None
    return value, (None if value is None else say(spec, value, full))


def _unmeasured_reason(record, flag, block, field, flag_state):
    """Why an item has no number: the flag is off, the block is absent, or the block is present
    and says why it scored nothing (reviews/dashboards.md M3)."""
    name = "--" + flag.replace("_", "-")
    b = (record or {}).get(block)
    if not isinstance(b, dict):
        if not flag_state.startswith("on"):
            return f"{name} is {flag_state}, so no record carries its measurement."
        return f"{name} is on but the newest record carries no '{block}' block."
    return f"the '{block}' block is present but carries no {field} this window" + (
        ": " + _block_counts(block, b) if _block_counts(block, b) else ".")


def _block_counts(block, b):
    """What a present block says about a window in which it scored nothing, in plain counts."""
    parts = []
    if block == "controllability":
        if _num(b.get("orders")) is not None:
            parts.append(f"{int(b['orders']):,} commands graded")
        if _num(b.get("measured_share")) is not None:
            parts.append(f"{per(b['measured_share'], 'commands')} scored (a command is scored only when the effect "
                         f"the model expected, and the command itself, rise above the body's drift while holding "
                         f"still, so stillness is never counted)")
        if _num(b.get("efference_missing")):
            parts.append(f"{int(b['efference_missing']):,} commands without the body head's command-long expectation")
        if _num(b.get("tenure_mismatch")):
            parts.append(f"{int(b['tenure_mismatch']):,} commands of the wrong length")
        if b.get("gate_open") is not None:
            # the producer's own reason, shown only when it passes DESIGN.md section 11 (its
            # "enough readings" form does; its numeric form prints raw agreements)
            why = printable(str(b.get("gate_why") or "")) if b.get("gate_why") else None
            parts.append("its evidence gate " + ("open" if b.get("gate_open") else "shut")
                         + (f" ({why})" if why else ""))
    elif block == "causal_chain":
        for k, noun in (("planner_events", "planning events"), ("events", "commands graded"),
                        ("latch_predictions", "commands predicted before they were sent")):
            if _num(b.get(k)) is not None:
                parts.append(f"{int(b[k]):,} {noun}")
    elif block == "contact_control":
        for k, noun in (("onsets_self", "new self-touches"), ("scorable_self", "scorable new self-touches"),
                        ("control_onsets_self", "onsets scored beside the control")):
            if _num(b.get(k)) is not None:
                parts.append(f"{int(b[k]):,} {noun}")
    elif block == "synergy":
        for k, noun in (("orders", "commands graded"), ("plans", "plans")):
            if _num(b.get(k)) is not None:
                parts.append(f"{int(b[k]):,} {noun}")
    elif block == "wellness_rank":
        if _num(b.get("planning_events")) is not None:
            parts.append(f"{int(b['planning_events']):,} planning events")
    return "; ".join(parts) + ("." if parts else "")


def _related_act(record):
    parts = []
    for sense, name in SENSES:
        v, _ = _say_field(f"act.{sense}", record)
        if v is not None:
            parts.append(f"{name} {say(SERIES[f'act.{sense}'], v, full=False)}")
    if not parts:
        return None
    return ("In the newest report window, knowing its own executed command improves its next-model-step prediction by "
            + ", ".join(parts) + " of the way from 'the command is not used' to 'exact prediction' (the control is "
                                 "another body's absolute command; one window is noisy; no validated success target).")


def _related_pers(record):
    parts = []
    for sense, name in SENSES:
        v, _ = _say_field(f"pers.{sense}", record)
        if v is not None:
            parts.append(f"{name} {say(SERIES[f'pers.{sense}'], v, full=False)}")
    if not parts:
        return None
    return ("It predicts the next model step better than \"nothing changes\" by " + ", ".join(parts)
            + " of the way to exact (below 0% = worse than predicting that nothing changes; no validated success target).")


def _related_follow(record):
    out = []
    ref = FOLLOW
    if ref.get("live") is not None:
        out.append(f"Counted on ckpt_step_344400 ({ref.get('measured')}): {fixed(ref['live'], 2)} of every 100 "
                   f"body-part commands ended closer to the commanded turn 250 ms later; the command ignored (chance) "
                   f"scores {fixed(ref['chance'], 2)}, 25 independent random turns on physics alone "
                   f"{fixed(ref['independent'], 2)}, a posture the body really reached on physics alone "
                   f"{fixed(ref['reachable'], 2)} (the goal line for following).")
    v, text = _say_field("follow", record)
    if v is not None:
        out.append("The live direction match, every command this window: " + text
                   + (f"; a reachable posture on physics alone reaches {say_ref(SERIES['follow'], FOLLOW['gauge_reachable'])}"
                      if FOLLOW.get("gauge_reachable") is not None else "") + " (not yet measured as a count per window).")
    return " ".join(out) or None


def _item_causal_chain(row, record, args):
    key, flag, block = "cc.chosen_best", "causal_chain", "causal_chain"
    b = (record or {}).get(block) if record else None
    best, _ = _say_field(key, record)
    lines = []
    if best is None:
        return None
    alt = _num(_g(b, "alternatives_n"))
    events = _num(_g(b, "planner_events"))
    chance = None if alt is None else 1.0 / (1.0 + alt)
    tied = _num(_g(b, "chosen_tied_share"))
    row["big"] = f"{per100_text(best)} in 100 plans"
    row["answer"] = "The planner aims toward a remembered state. This check asks a different question: did it choose the movement predicted to improve its condition most? Useful choices have not been established."
    row["beats"] = False
    lines.append(_line("measure", f"The plan it chose was the one it predicted would feel best in "
                                  f"{per(best, 'plans')}" + (f" ({int(events):,} planning events this window)"
                                                              if events is not None else "") + "."))
    if chance is not None:
        lines.append(_line("control", f"No random-choice comparison was measured for this window. Picking at random among the {int(alt) + 1} candidates would choose the best in "
                                      f"{per(chance, 'plans')} only if exactly one option were distinguishably best. Ties invalidate that comparison."))
    if tied is not None:
        lines.append(_line("control", f"Tied in {per(tied, 'plans')}" + tie_meaning(b)
                                      + "; a tie is not counted as choosing the best."))
    else:
        lines.append(_line("control", "Ties are not reported by this record (no chosen_tied_share): a plan tied with "
                                      "an alternative counts as the best here."))
    gap = _num(_g(b, "gap_to_best_alt_mean"))
    spread = _num(_g(b, "candidate_spread_mean"))
    if gap is not None or spread is not None:
        bits = []
        if gap is not None:
            bits.append(f"the chosen movement’s predicted condition was {'below' if gap < 0 else 'above'} the best alternative by {say(SERIES['cc.gap'], abs(gap), full=False)} of the full score range (0 to 5)")
        if spread is not None:
            bits.append(f"the candidates spread over {say(SERIES['cc.spread'], spread, full=False)} of it")
        lines.append(_line("measure", "On average " + " and ".join(bits) + "."))
    else:
        pc, pa = _num(_g(b, "pred_wellness_chosen")), _num(_g(b, "pred_wellness_best_alt"))
        if pc is not None and pa is not None:
            lines.append(_line("measure", f"The window's mean predicted wellness of the chosen plans and of the best "
                                          f"alternatives differ by {say(SERIES['cc.gap'], abs(pc - pa), full=False)} "
                                          f"of the evaluator's 0-to-5 scale (no per-plan gap in this record)."))
    dist = _num(_g(b, "plan_exec_distance_cm"))
    if dist is not None:
        lines.append(_line("measure", f"The command the body received sat {fixed(dist, 2)} cm from the command the "
                                      f"planner imagined sending (0 cm = the body got what was imagined)."))
    lines.append(_line("goal", "No validated success target: " + GOAL_WHY_CHAIN + ". Deliberate useful movement still needs a controlled test."))
    row["lines"] = lines
    # not a share of the way from knowing nothing: picking at random scores 1 in (1 + alternatives),
    # and an evaluator that ties every candidate scores 100 (reviews/dashboards.md M1)
    row["share"] = None
    return row


def per100_text(share):
    return f"{100.0 * float(share):,.0f}" if abs(100.0 * float(share)) >= 1 or share == 0 else fixed(100.0 * float(share), 2)


def _against_control(row, value, control, gate, control_name, gate_name):
    """The answer of an item measured against a control, and whether it BEATS the control.

    A reading above its control by any amount is not a win: agreement 0.011 against a control of
    0.010 sits inside the control's own spread (reviews/dashboards.md B1: the control read 0.005 to
    0.017 in b3's measurement 2). An item beats its control only when the producer's own evidence
    test for that comparison is open (`gate`: controllability.gate_open, a mean difference above
    the recorded noise threshold over the last 32 readings; contact_control.gate_self_open, the
    same test on new self-touches). Without an evidence test the difference is reported and not
    counted (verify-fix-dashboards, 2026-09-23)."""
    row["beats"] = False
    if control is None:
        row["answer"] = "No matching comparison was recorded, so an advantage cannot be assessed."
    elif value <= control:
        row["answer"] = f"No advantage over {control_name} in this report."
    elif gate is True:
        row["beats"] = True
        row["answer"] = f"Better than {control_name}; the recent statistical check supports the difference. This does not establish a useful skill."
    elif gate is False:
        row["answer"] = f"A benefit over {control_name} has not been established by the recent statistical check."
    else:
        row["answer"] = f"The score is higher than {control_name}, but no statistical check was recorded for this difference."
    return row["beats"]


def _item_controllability(row, record, args):
    b = (record or {}).get("controllability")
    agree, _ = _say_field("ctl.agreement", record)
    if agree is None:
        return None
    spec = SERIES["ctl.agreement"]
    ctl = _num(_g(b, "control_mean"))
    diff = _num(_g(b, "difference_mean"))
    row["big"] = say(spec, agree, full=False)
    row["share"] = None          # set against its control, never against "knowing nothing" (B1)
    lines = [_line("measure", "Agreement between predicted and actual movement direction and size: "
                              + say(spec, agree) + " This is an average agreement score, not a success rate.")]
    if ctl is not None:
        lines.append(_line("control", "The same body's result against another body's command (the control): "
                                      + say(spec, ctl, full=False) + (f"; the difference {say(spec, diff, full=False)}"
                                                                     if diff is not None else "") + "."))
    else:
        lines.append(_line("control", "No control this window (it needs more than one body): the reading cannot be "
                                      "told from an expectation that ignores the command."))
    ms = _num(_g(b, "measured_share"))
    if ms is not None:
        lines.append(_line("measure", f"Scored on {per(ms, 'commands')} (a command is scored only when an effect was "
                                      f"expected and commanded larger than movement measured while holding still)."))
    if b and b.get('gate_open') is not None:
        allowed = 'may now influence' if b.get('gate_open') else 'does not currently influence'
        lines.append(_line('control', f'The recent statistical check says this command-prediction score {allowed} which remembered state is chosen as a goal. It uses recent scored readings, a different window from the average above.'))
        thresholds = b.get('gate_lines_noise') or {}
        if _num(thresholds.get('open')) is not None and _num(thresholds.get('close')) is not None:
            lines.append(_line('source', f"The score is enabled when its advantage exceeds {fixed(thresholds['open'], 1)} times its estimated measurement noise; once enabled it is disabled below {fixed(thresholds['close'], 1)} time. Simultaneous bodies share a model and are not independent learning trials."))
    # the gate in force beside the recorded-only shadow gate, on the same graded commands (2026-10-03)
    gates = controllability_gates_text(b)
    if gates:
        lines.append(_line('control', gates))
    _against_control(row, agree, ctl, b.get('gate_open') if isinstance(b, dict) else None,
                     'a prediction from the same body state using another body’s command', 'the recent statistical check')
    lines.append(_line("goal", "No success threshold has been validated for this measurement."))
    row["lines"] = lines
    return row


def _item_learning_progress(row, record, args):
    b = (record or {}).get("learning_progress")
    v, _ = _say_field("lp.value", record)
    if v is None:
        return None
    spec = SERIES["lp.value"]
    row["big"] = say(spec, v)
    row["share"] = None
    lines = [_line("measure", f"Prediction error compared with predicting no change: {say(spec, v)} "
                              f"compared with its longer-term average. Positive means recent errors fell; negative means they rose.")]
    per_sense = _g(b, "per_sense")
    if isinstance(per_sense, dict):
        names = dict(ALL_SENSES)
        bits = [f"{names.get(k, k.replace('_', ' '))} {say(spec, _num(x))}" for k, x in per_sense.items()
                if _num(x) is not None]
        if bits:
            lines.append(_line("measure", "Per sense: " + ", ".join(bits) + "."))
    ctl = _num(_g(b, "control_mean"))
    lo, hi = LP_FROZEN_BAND
    if ctl is not None:
        lines.append(_line("control", f"The same-model-step control read {say(spec, ctl)}."))
        inside = None
        # no noise line for this difference exists yet: a producer evidence test (gate_open) is
        # read when the block carries one
        _against_control(row, v, ctl, b.get("gate_open") if isinstance(b, dict) else None,
                         "its same-model-step control", "its evidence gate")
    else:
        # the band's own width is the spread the world's changes alone produced on frozen weights
        # (three windows, b3 measurement 2): a reading above the band by less than that width is
        # not told apart from it (reviews/dashboards.md B1 (b): training read 0.006 to 0.012)
        inside = lo <= v <= hi
        width = 100.0 * (hi - lo)
        above = 100.0 * (v - hi)
        lines.append(_line("control", f"No matched comparison yet. An earlier experiment with learning disabled "
                                      f"produced scores from {say(spec, lo)} to {say(spec, hi)} as its experiences changed (2026-09-22). This historical range is not a matched comparison for the current run; "
                                      + ("this reading is inside that band, so it is not evidence of learning."
                                         if inside else
                                         "this reading is below that band." if v < lo else
                                         f"this reading is {fixed(above, 2)} percentage points above that band, whose "
                                         f"own width is {fixed(width, 2)}.")))
        row['answer'] = 'Learning is not established by this score. Changing surroundings can also change prediction errors. Earlier and later models need the same test examples.'
        row["beats"] = False
    lines.append(_line("goal", "No success threshold has been validated for this measurement."))
    row["lines"] = lines
    return row


def _item_wellness_rank(row, record, args):
    b = (record or {}).get("wellness_rank")
    lc = lead_channel(record) if record else None
    if lc is None:
        return None
    name, lead, weight = lc
    row["big"] = f"Strongest influence: {name.replace(chr(95), chr(32))}"
    row["share"] = None
    lines = [_line("measure", f"A goal's advantage is its total score minus the next-best goal's score. The largest contribution to that advantage comes from "
                              f"{name.replace('_', ' ')}: {fixed(100 * lead, 2)} of every 100 parts of the lead"
                              + (f", while it holds {fixed(100 * weight, 2)} of every 100 parts of the ranking's "
                                 f"weights" if weight is not None else ", its share of the weights not in this record")
                              + ". The channels' parts of the lead add to 100 and can be below 0 or above 100.")]
    ga = _g(b, "goal_age")
    if isinstance(ga, dict) and _num(ga.get("mean_share_back")) is not None:
        ev = _num(ga.get("events"))
        old, new = _num(ga.get("oldest_tenth_share")), _num(ga.get("newest_tenth_share"))
        lines.append(_line("measure", f"The chosen goal sat on average {pct(ga['mean_share_back'], 2)} of the way back "
                                      f"through the list of remembered body states (0% = the newest remembered model step, 100% = the oldest, "
                                      f"10 s back)" + (f" over {int(ev):,} choices" if ev is not None else "") + "."))
        if old is not None or new is not None:
            lines.append(_line("control", "Goals from the oldest tenth of those memories: "
                                          + (per(old, 'goals') if old is not None else "unmeasured")
                                          + "; from the newest tenth: "
                                          + (per(new, 'goals') if new is not None else "unmeasured")
                                          + "; a choice blind to age puts 10 of every 100 in each tenth."))
        row["answer"] = ("Choices concentrate on older or newer experiences. A controlled test is needed to find out whether age causes this preference."
                         if (old is not None and old > 0.5) or (new is not None and new > 0.5) else
                         "Choices are recorded, but we have not shown that they lead to better outcomes.")
    else:
        lines.append(_line("control", "Goal age is not in this record (wellness_rank.goal_age absent): whether goals are "
                                      "chosen by how old a model step is cannot be checked (b3 review: 42 of 44 goals were "
                                      "the oldest model steps in the build's own run)."))
        row["answer"] = "The ages of the chosen memories are missing, so an age preference cannot be checked."
    gated = _num(_g(b, "integrity_gated_share"))
    if gated is not None:
        lines.append(_line("measure", f"The harm check excluded {per(gated, 'candidate model steps')}."))
    lines.append(_line("control", "Before normalisation, the raw ranking's consequence term carried 96 of every 100 "
                                  "parts of the goal value's movement (DESIGN.md section 0)."))
    lines.append(_line("goal", "No success threshold has been validated for this measurement."))
    row["lines"] = lines
    return row


def _physics_follow_text(spec):
    """The physics-only references of following in one sentence, the part count and angle read from the
    stored reference (never a fixed "25 independent 20-degree turns")."""
    text = "On physics alone"
    if FOLLOW.get("gauge_independent") is not None and FOLLOW_TURNS:
        text += f", {FOLLOW_TURNS}, reach {say_ref(spec, FOLLOW['gauge_independent'])} and"
    return text + f" commands to a posture the body really reached reach {say_ref(spec, FOLLOW['gauge_reachable'])}"


def synergy_outputs_used(record, args):
    """Which of the motor synergy's outputs wrote executed orders in this record, or None when the record
    counts no executed orders by author (synergy.following_by_author), so only the launch settings can
    describe the run. Item 5 and the Now summary (training_brief.assessment, whose detail item 5's card
    prints) both read it, so the two never disagree on one card.

    These are actual executed orders; a planner entry is not automatically a synergy order. Whether the
    synergy wrote any of them is read from the record's own counts (lem_model.synergy_window_summary):
    plans, the planner decisions made through the synergy, and babble_steps, the moments of exploration
    drawn in it, each beside the executed orders of the same author. The launch flags say only which
    outputs were allowed (the style guide: a flag state is never the evidence that an effect exists); one
    decides only for a record without its count (every record since 80eaf88ac carries both), read at the
    trainer's default when the key is missing too (train_mimo.parse_args: --synergy-planner on,
    --synergy-babble off)."""
    b = (record or {}).get("synergy")
    by = _g(b, "following_by_author")
    counts = {k for k, v in (by or {}).items() if isinstance(v, dict) and (_num(v.get("part_orders")) or 0) > 0}
    if not counts:
        return None
    args = args or {}
    plans, babble_steps = _num(_g(b, "plans")), _num(_g(b, "babble_steps"))
    planner_on = plans > 0 if plans is not None else args.get("synergy_planner", "on") == "on"
    babble_on = babble_steps > 0 if babble_steps is not None else args.get("synergy_babble", "off") == "on"
    return dict(planning=planner_on and "planner" in counts, exploration=babble_on and "reflex_noise" in counts,
                plans=plans, babble_steps=babble_steps)


def _item_synergy(row, record, args):
    b = (record or {}).get("synergy")
    by = _g(b, "following_by_author")
    counts = {k: v for k, v in (by or {}).items()
              if isinstance(v, dict) and (_num(v.get("part_orders")) or 0) > 0}
    if counts:
        outputs = synergy_outputs_used(record, args)
        plans, babble_steps = outputs["plans"], outputs["babble_steps"]
        used = [k for k, on in (("planner", outputs["planning"]), ("reflex_noise", outputs["exploration"])) if on]
        row["stand_in"] = True
        row["big"] = "Control comparison pending" if used else "Learning from movement; not directing it"
        row["answer"] = ("Following is counted below, but improvement from the synergy needs a paired "
                         "comparison with the same controller and starts."
                         if used else "The coordination network learns from completed movements. No commands from it "
                         "were used in this report. Its effect needs the same tests with its control outputs enabled and disabled.")
        row["share"] = None
        lines = [_line('measure', 'The counts below describe commands actually used by the current controller. They do not establish an improvement from synergy control.')]
        if plans is not None and babble_steps is not None:
            lines.append(_line("measure", f"In this report the planner planned through the synergy {int(plans):,} "
                               f"times and exploration was drawn from it at {int(babble_steps):,} model steps (fields "
                               f"synergy.plans and synergy.babble_steps)."))
        else:
            lines.append(_line("measure", "This report does not count the commands the synergy formed, so the launch "
                               "settings decide whether they were used."))
        for author, c in sorted(counts.items()):
            n, closer, chance = c.get("part_orders"), c.get("closer"), c.get("chance_count")
            if any(_num(v) is None for v in (n, closer, chance)):
                continue
            author_name = {'external_rest':'scheduled hold/rest', 'reflex_noise':'motor controller with exploration', 'reflex':'motor controller without exploration', 'planner':'planner'}.get(author, author.replace('_', ' '))
            lines.append(_line("measure", f"{author_name}: {int(closer)} of {int(n)} body-part "
                               f"commands ended closer, against {fixed(chance, 1)} expected by the random-axis "
                               f"control; {fixed(100 * (closer - chance) / n, 1)} more per 100 commands."))
            moved, asked = c.get("moved_deg_median"), c.get("ordered_deg_median")
            if _num(moved) is not None and _num(asked) is not None:
                lines.append(_line("related", f"{author_name}: median realized turn "
                                   f"{fixed(moved, 1)} degrees; median commanded turn {fixed(asked, 1)} degrees."))
        lines.append(_line("control", "Chance changes the rotation axis at the same commanded angle against "
                           "the observed turn. It is a geometric control, not a physically replayed command. "
                           "Stillness earns no following credit. Faster learning remains unmeasured."))
        row["lines"] = lines
        return row
    v = _synergy_following(record) if record else None
    every = _num(_g(b, "following_mean"))
    spec = SERIES["syn.following"]
    row["share"] = None
    if v is None and every is not None:
        # every graded order's following (babble and reflex orders included) is a stand-in, shown
        # and not counted, as the posture probe is for item 10 (reviews/dashboards.md M2, M5)
        ps = _num(_g(record, "exec_source", "planner"))
        row["stand_in"] = True
        row["big"] = "Indirect test only"
        row["answer"] = ("Unmeasured: the synergy block does not split following by who wrote the command "
                         "(synergy.following_by_author absent), so the commands the synergy map writes are not scored "
                         "apart. A stand-in reading exists (below); it is not counted as this item's measurement.")
        lines = [_line("related", "Stand-in reading, not this item's measurement: the body followed every command graded "
                                  "this window, whoever wrote it: " + say(spec, every) + " (field synergy.following_mean; "
                                  + (f"the planner, whose commands the synergy writes, wrote {per(ps, 'latched commands')}"
                                     if ps is not None else "who wrote the commands is not in this record")
                                  + "; not yet measured as a count).")]
        if FOLLOW.get("gauge_reachable") is not None:
            lines.append(_line("control", _physics_follow_text(spec) + "."))
            lines.append(_line("goal", "Goal line: the reachable posture's following on physics alone; distance to goal "
                                       "unmeasured until the synergy's own commands are scored."))
        else:
            lines.append(_line("goal", "No validated success target: " + str(FOLLOW_GOAL_WHY) + "; distance to goal "
                                       "unmeasured."))
        row["lines"] = lines
        return row
    if v is None:
        return None
    which = "the planner's synergy-written commands"
    row["big"] = say(spec, v, full=False)
    lines = [_line("measure", f"The body followed {which}: {say(spec, v)} (a direction match over each command's "
                              f"250 ms, averaged; field synergy.following_by_author; not yet measured as a count).")]
    if every is not None:
        lines.append(_line("related", f"Every command graded this window, whoever wrote it: {say(spec, every, full=False)}."))
    if FOLLOW.get("gauge_reachable") is not None:
        lines.append(_line("control", _physics_follow_text(spec) + f" ({FOLLOW.get('gauge_source')})."))
        gap = shown(spec["display"], FOLLOW["gauge_reachable"]) - shown(spec["display"], v)
        lines.append(_line("goal", f"Goal line: the reachable posture's {say_ref(spec, FOLLOW['gauge_reachable'])}; "
                                   f"distance {fixed(abs(gap), 2)} percentage points {'short of' if gap > 0 else 'past'} it."))
        row["answer"] = ("Not shown: followed no better than independent turns on physics alone."
                         if FOLLOW.get("gauge_independent") is not None and v <= FOLLOW["gauge_independent"] else
                         "Short of a reachable posture's following." if gap > 0 else "At or past the goal line.")
    else:
        lines.append(_line("goal", "No validated success target: " + str(FOLLOW_GOAL_WHY) + "; distance from a working skill has not been measured."))
        row["answer"] = "The prediction is measured, but no test has established deliberate posture control."
    row["lines"] = lines
    return row


def _item_contact(row, record, args):
    b = (record or {}).get("contact_control")
    if not isinstance(b, dict):
        return None
    auc, ctl = _num(b.get("onset_auc_self")), _num(b.get("control_onset_auc_self"))
    paired, sctl = _num(b.get("success_paired")), _num(b.get("success_control"))
    # the self-touch evidence gate is the producer's test on new self-touches alone; the older
    # pooled gate (gate_open) opens on touches by the world too (reviews/b4-self-contact.md M1),
    # so it is not evidence about self-touch
    gate_self = b.get("gate_self_open") if isinstance(b.get("gate_self_open"), bool) else None
    gate_name = "its self-touch evidence gate"
    lines = []
    if auc is not None:
        row["big"] = f"{per100_text(auc)}% touch-ranking accuracy"
        lines.append(_line("measure", f"Ranking accuracy: how often a skin area about to be touched scores higher than one that stays untouched. Among areas untouched the model step before, it gave the area about to be touched "
                                      f"by its own body the higher chance in {per(auc, 'pairs')} (a coin flip scores 50; "
                                      f"field onset_auc_self)."))
        if ctl is not None:
            lines.append(_line("control", f"A base-rate guess that knows only each patch's own history scored "
                                          f"{per(ctl, 'pairs')} on the same skin areas at the same times (control_onset_auc_self)."))
        else:
            lines.append(_line("control", "The base-rate control is not in this record (control_onset_auc_self absent)."))
        _against_control(row, auc, ctl, gate_self, "a base-rate guess", gate_name)
    elif paired is not None:
        row["big"] = f"{per100_text(paired)} of 100 new self-touches predicted"
        n = _num(b.get("control_onsets_self"))
        lines.append(_line("measure", f"It foresaw {per(paired, 'new self-touches')} with its own command"
                                      + (f" ({int(n):,} onsets scored beside the control)" if n is not None else "")
                                      + " (field success_paired; the onset score onset_auc_self is not in this record)."))
        if sctl is not None:
            lines.append(_line("control", f"On the same onsets, another body's command foresaw "
                                          f"{per(sctl, 'new self-touches')} (success_control)."))
        _against_control(row, paired, sctl, gate_self, "another body's command on the same onsets", gate_name)
    else:
        return None
    # not a share of the way from knowing nothing: a coin flip scores 50 on the onset score, and
    # both readings are set against their controls (reviews/dashboards.md B1)
    row["share"] = None
    if b.get("gate_self_open") is not None or b.get("gate_world_open") is not None:
        lines.append(_line("measure", "May predicted touch influence goal choice? Own body: " + ("yes" if b.get("gate_self_open") else "no")
                                      + "; surroundings: " + ("yes" if b.get("gate_world_open") else "no") + "."))
    elif b.get("gate_open") is not None:
        lines.append(_line("measure", "The combined check for touch of the body and surroundings "
                                      + ("allows" if b.get("gate_open") else "does not allow") + " predicted touch to influence goal choice."))
    hab = _num(b.get("habituation_mean"))
    if hab is not None:
        lines.append(_line("measure", f"Repeated contacts have less influence on goal choice: the current multiplier averages {pct(hab, 2)} of a new contact’s influence. 100% = a new pair of touching parts; lower values mean repeated contact."))
    head_auc = _num(b.get("head_auc"))
    if head_auc is not None:
        lines.append(_line("related", f"Any-contact score over every patch at every model step, persisting contacts included: "
                                      f"{per(head_auc, 'pairs')} (mostly measures whether a touch continues; not this "
                                      f"item's measurement)."))
    lines.append(_line("goal", "No success threshold has been validated for this measurement."))
    row["lines"] = lines
    return row


def _item_vestibular(row, record, args):
    v, text = _say_field("vest.gain", record)
    if v is None:
        return None
    row["big"] = say(SERIES["vest.gain"], v, full=False)
    row["share"] = v
    row["lines"] = [_line("measure", "The vestibular input is predicted " + text + "."),
                    _line("control", "The floor is predicting that nothing changes (0%)."),
                    _line("goal", "No success threshold has been validated for this measurement.")]
    row["answer"] = "The prediction is measured, but no test has established deliberate posture control."
    return row


# Each item: the direct measurement (a builder over its section 3 block, or a probe item), the
# flag that turns it on, and the reading already logged beside it (never offered as the item's
# own measurement).
ITEMS = [
    dict(item=1, name="Choosing a useful movement", question="Does it choose the action it predicts will feel best?",
         key="cc.chosen_best", flag="causal_chain", block="causal_chain", build=_item_causal_chain,
         related=_related_act,
         goal_how="needs a planner that aims at predicted wellness, then its choice rate on candidates the "
                  "evaluator can tell apart"),
    dict(item=2, name="Predicting the effect of a command", question="Does the body do what it expected its command to do?",
         key="ctl.agreement", flag="controllability", block="controllability", build=_item_controllability,
         related=_related_follow,
         goal_how="the agreement a physics-only reference predictor reaches on the same commands"),
    dict(item=3, name="Evidence of learning", question="Are its prediction errors falling?",
         key="lp.value", flag="learning_progress", block="learning_progress", build=_item_learning_progress,
         related=_related_pers, goal_how="the progress a head trained on recorded model steps reaches, against its frozen copy"),
    dict(item=4, name="What drives its choices?",
         question="Is the goal picked by one wellness channel only because its numbers are larger?",
         key="wr.lead", flag="wellness_rank", block="wellness_rank", build=_item_wellness_rank, related=None,
         goal_how="each channel's lead share against its weight share on model steps with no age trend"),
    dict(item=5, name="Learning coordinated movements", question="Does the body follow the commands the synergy map writes?",
         key="syn.following", flag="motor_synergy", block="synergy", build=_item_synergy,
         related=lambda r: (lambda v: None if v is None else
                            "The map reproduces " + say(SERIES["syn.explained"], v)
                            + " of the posture changes the body made, before learning from them (can be below 0; "
                              "high while controlling nothing if it reproduces the average change).")(
             _num(_g(r, "synergy", "explained_share"))),
         goal_how="compare synergy control enabled and disabled from identical starting states; count the commands after which the body ends nearer its target than it started, then repeat on new states"),
    dict(item=6, name="Predicting new self-touch", question="Is a touch of its own body foreseen before it happens?",
         key="ccx.onset_self", flag="contact_head", block="contact_control", build=_item_contact,
         related=lambda r: (lambda v: None if v is None else
                            "For bodies touching themselves, knowing the actual command reduces the tactile prediction's error by "
                            + say(SERIES["self_touch_act"], v) + " compared with another body's command. Below 0% means more error. This includes floor contact and does not isolate self-touch.")(
             _num(_g(r, "self_touch_controlled", "improvement_over_shuffled"))),
         goal_how="the onset score a reference predictor fitted on recorded commands and contacts reaches"),
    dict(item=7, name="Predicting eye-movement consequences", question="Does changing only the eye command shift the predicted view?",
         probe=7, related=None, goal_how="the share a small reference predictor fitted on this world's recorded "
                                        "commands and outcomes reaches"),
    dict(item=8, name="Predicting limb-movement consequences", question="Do different commands give different, sensible predicted outcomes?",
         probe=8, related=None, goal_how="a physics-only ceiling: each body part's change predicted from the command "
                                         "through the decoder on physics alone"),
    dict(item=9, name="Predicting hand movement and touch",
         question="Does the predicted arm movement match what it sees and feels?", probe=9, related=None,
         goal_how="the share a reference predictor fitted on recorded commands and views reaches"),
    dict(item=10, name="Predicting posture and balance", question="Does it predict its own posture and balance?",
         key="vest.gain", flag="vestibular_sense", block="vestibular", build=_item_vestibular, probe=10,
         related=lambda r: (lambda rolls: None if not rolls else
                            f"How far the bodies are turned from lying on the back: {fixed(sum(rolls) / len(rolls), 2)} "
                            f"degrees on average, {fixed(max(rolls), 2)} degrees for the most-turned body (0 = on its "
                            f"back, 90 = on its side, 180 = belly down; no validated success target).")(_rolls(r or {})),
         goal_how="vestibular's error against a physics-only predictor of the same fields"),
]


def _probe_lines(entries):
    lines = []
    for e in entries:
        w = e.get("words") or probe_words(e.get("latest"), e.get("probe"))
        step = (e.get("latest") or {}).get("lineage_step")
        lines.append(_line("probe", w["sentence"] + (f" (lineage step {int(step):,})" if step is not None else "")))
        if w.get("scale"):
            lines.append(_line("scale", w["scale"]))
        lines.append(_line("goal", w["goal"]))
        if e.get("failed"):
            lines.append(_line("failed", e["failed"].get("text") or failed_probe_text(e)))
        if w.get("own"):
            lines.append(_line("details", "The probe's own summary: " + w["own"]))
        elif w.get("own_withheld"):
            lines.append(_line("details", "The probe's own summary carries numbers in a form this page does not print; "
                                          "read it in " + str((e.get("latest") or {}).get("path") or "its result file") + "."))
    return lines


def _efference_beside(item, record):
    """Readings of the efference heads shown beside a probe item, never in its place."""
    ef = (record or {}).get("efference")
    if not isinstance(ef, dict):
        return None
    if item == 7:
        e, d = _list_item(("efference", "gaze_deg_err"), 2)(record), _list_item(("efference", "gaze_deg_do_nothing"), 2)(record)
        if e is None or d is None:
            return None
        return (f"Beside it, eye angles alone: the efference gaze head's eye-angle error 150 ms ahead is {fixed(e, 2)} "
                f"degrees against {fixed(d, 2)} degrees for predicting the eyes stay. That is where the eyes point, not "
                f"what they will see; the probe reading above is what they see.")
    if item == 8:
        e, d = _num(ef.get("body_cm_err_latch")), _num(ef.get("body_cm_do_nothing_latch"))
        s = _num(ef.get("body_cm_shuffled_latch"))
        if e is None or d is None:
            return None
        return (f"Beside it, the efference body head: over a command it misplaces each body part by {fixed(e, 2)} cm, "
                f"against {fixed(d, 2)} cm for predicting no movement" + (f" and {fixed(s, 2)} cm with another body's "
                                                                         f"command" if s is not None else "") + ".")
    return None


def gain_of_function(record, args, probes):
    """The Now tab's ten items. Each row carries the sentences the page prints (`lines`), built
    here so the page does no string building of its own: the answer first, the measurement with
    its range, its control, its goal line and distance (or the words "unmeasured"), and the reading
    already logged beside it, labelled as not the item's measurement."""
    record, args = record or {}, args or {}
    items = (probes or {}).get("items", {})
    out = []
    for spec_item in ITEMS:
        row = dict(item=spec_item["item"], name=spec_item["name"], question=spec_item["question"],
                   state="unmeasured", measured=False, value=None, big="Unmeasured", answer=None, lines=[],
                   share=None, goal_how=spec_item.get("goal_how"))
        key = spec_item.get("key")
        if key:
            flag = spec_item["flag"]
            row.update(flag="--" + flag.replace("_", "-"), flag_state=_flag_state(args, flag),
                       metric=SERIES[key]["title"], field=SERIES[key].get("field"))
            built = spec_item["build"](row, record, args) if record else None
            if built is not None and built.get("stand_in"):
                # a builder that found only a stand-in reading (item 5's every-order following):
                # shown, never counted as the item's measurement
                row.update(state="stand-in", measured=False, source="newest training record (stand-in)")
            elif built is not None:
                row.update(state="measured", measured=True, source="newest training record")
                try:
                    row["value"] = SERIES[key]["get"](record)
                except (TypeError, ValueError, ZeroDivisionError):
                    row["value"] = None
            else:
                row["lines"] = []
                reason = _unmeasured_reason(record, flag, spec_item["block"], SERIES[key].get("field") or key,
                                            row["flag_state"])
                row["unmeasured_because"] = reason[0].upper() + reason[1:]
                row["lines"].append(_line("source", row["unmeasured_because"]))
                row["answer"] = ('The feature is enabled, but this report contains no usable measurement.' if str(row.get('flag_state', '')).startswith('on') else 'This feature is disabled or was not configured at launch.')
        probe_item = spec_item.get("probe")
        if probe_item is not None:
            all_entries = items.get(str(probe_item)) or []
            entries = [e for e in all_entries if e.get("latest")]
            # a probe whose runs on this lineage produced no reading (failure records only)
            failed_only = [e for e in all_entries if not e.get("latest") and e.get("failed")]
            for e in entries:
                if not e.get("words"):
                    e["words"] = probe_words(e.get("latest"), e.get("probe"))
            direct = [e for e in entries if not e["words"]["stand_in"]]
            stand = [e for e in entries if e["words"]["stand_in"]]
            row["probes"] = [dict(probe=e.get("probe"), value=_num(e["latest"].get("value")),
                                  step=e["latest"].get("lineage_step"), label=e["words"]["label"],
                                  stand_in=e["words"]["stand_in"]) for e in entries]
            if entries:
                latest_step = int(max((e['latest'].get('lineage_step') or 0) for e in entries))
                m = re.search(r'ckpt_step_(\d+)', str(args.get('resume') or ''))
                before = bool(m and latest_step <= int(m[1]))
                row['test_scope'] = (f'Last saved-model test: update {latest_step:,}. ' +
                                     ('It predates this restart; the current version has not been tested here.' if before else
                                      'This is a saved-model result, not a test of the changing live model.'))
            if not row["measured"] and direct:
                first = direct[0]
                w = first["words"]
                row.update(state="measured", measured=True, value=_num(first["latest"].get("value")),
                           source="probe index: " + str(first.get("probe")), big=w["big"], share=w["share"],
                           answer='This measures how command changes affect predictions. Repeatable useful movement has not been tested here.')
                row["lines"] = _probe_lines(direct)
            elif direct:
                row["lines"] += _probe_lines(direct)
            if stand:
                if not row["measured"]:
                    row["state"] = "stand-in"
                    row["big"] = "Indirect test only"
                    row["answer"] = ((row.get("answer") or "Unmeasured.") + " A stand-in reading exists (below); it "
                                     "is not counted as this item's measurement.")
                row["lines"] += [_line("related", "Stand-in reading, not this item's measurement:")] + _probe_lines(stand)
            fails = [e["failed"].get("text") or failed_probe_text(e) for e in failed_only]
            if fails and not entries and not key:
                reason = " ".join(fails)
                row["unmeasured_because"] = reason
                row["answer"] = "The test stopped with an error; no result is available."
                row["lines"].append(_line("source", reason))
            elif fails:
                row["lines"] += [_line("failed", t) for t in fails]
            elif not entries and not key:
                row["unmeasured_because"] = ("No probe result on this model's ancestry yet; the probe suite writes "
                                             "Saved/mimo_bridge/gof_probes/index.jsonl."
                                             if not (probes or {}).get("index_found") else
                                             "The probe index holds no result for this item on this model's ancestry.")
                row["answer"] = "Unmeasured: " + row["unmeasured_because"][0].lower() + row["unmeasured_because"][1:]
            beside = _efference_beside(probe_item, record)
            if beside:
                row["lines"].append(_line("related", beside))
        rel = spec_item.get("related")
        row["related"] = rel(record) if rel and record else None
        if row["related"]:
            row["lines"].append(_line("related", "Already logged beside it (not this item's measurement): "
                                      + row["related"]))
        if row.get("field") and row["measured"] and row.get("source") == "newest training record":
            row["lines"].append(_line("source", "Field read: " + row["field"] + "."))
        if row.get("goal_how"):
            row["lines"].append(_line("how", "What would establish improvement: " + row["goal_how"] + "."))
        out.append(row)
    measured = sum(1 for r in out if r["state"] == "measured")
    stand = sum(1 for r in out if r["state"] == "stand-in")
    return dict(items=out, measured=measured, stand_in=stand, total=len(out))


def item_text(row):
    """Every sentence one Now card prints, in order: what the page shows for the item (the test
    renders this and the page renders the same fields)."""
    parts = [f"{row['item']}. {row['name']}", row["question"], row.get("big") or "", row.get("answer") or ""]
    parts += [ln["text"] for ln in row.get("lines") or []]
    return "\n".join(p for p in parts if p)


def headline(panel, record, action_use=None):
    """The Now tab's generated headline: the abilities first (reviews/dashboards.md M6), then the
    instruments, then the action-use reading with a section-20 sentence that depends on it:
    `action_use` is action_use_summary()'s result (the newest segment's median against the
    pre-build windows' middle 90 in 100); without it, the newest window is quoted and no line is
    drawn."""
    items = panel["items"]
    direct = [r for r in items if r["state"] == "measured"]
    stand = [r for r in items if r["state"] == "stand-in"]
    unmeasured = [r for r in items if r["state"] == "unmeasured"]
    # Only the probe readings share one scale whose low end is knowing nothing ("the prediction
    # ignores the order"); a training-record item is judged against its own control, never placed
    # against knowing nothing here (the onset score's knowing-nothing is a coin flip, 50, and an
    # evaluator that ties every candidate puts item 1 at 100: reviews/dashboards.md B1, M1, M6).
    probe_rows = [r for r in direct if str(r.get("source") or "").startswith("probe index")
                  and r.get("share") is not None]
    beating = [r for r in direct if r.get("beats")]
    reached = [r for r in direct if (r.get("answer") or "").startswith("At or past")]
    if reached:
        title = (f"{len(reached)} item{'s reach their' if len(reached) > 1 else ' reaches its'} measured goal line "
                 f"in the newest window ({', '.join(r['name'].lower() for r in reached)}); one window is not a "
                 f"lasting ability.")
    elif beating:
        title = (f"{len(beating)} item{'s beat their controls' if len(beating) > 1 else ' beats its control'} by the "
                 f"producer's own evidence test; none has a measured goal line to reach yet.")
    else:
        title = "No gain-of-function item is shown working yet: none has a measured goal line to reach."
    bits = []
    if direct:
        names = ", ".join(r["name"].lower() for r in direct)
        bits.append(f"{len(direct)} of {panel['total']} items {'have' if len(direct) > 1 else 'has'} a direct "
                    f"measurement ({names}).")
        judged = [r for r in direct if r not in probe_rows and r.get("beats") is not None]
        if judged:
            bits.append(f"{len(beating)} of the {len(judged)} item{'s' if len(judged) > 1 else ''} measured against a "
                        f"control {'beats' if len(beating) == 1 else 'beat'} it by the producer's own evidence test"
                        + (f" ({', '.join(r['name'].lower() for r in beating)})" if beating else "") + ".")
        if probe_rows:
            top = max(probe_rows, key=lambda r: r["share"])
            bits.append(f"The largest probe reading is {pct(top['share'], 2)} of the way from 'the prediction ignores "
                        f"the command' to 'it foresees the whole change' ({top['name'].lower()}).")
    else:
        bits.append(f"None of the {panel['total']} items has a direct measurement on the running model yet.")
    if stand:
        bits.append(f"{len(stand)} item{'s have' if len(stand) > 1 else ' has'} only a stand-in reading, not counted.")
    if unmeasured:
        bits.append(f"{len(unmeasured)} item{'s are' if len(unmeasured) > 1 else ' is'} unmeasured.")
    names = dict(SENSES)
    use = {s: v for s, v in (action_use or {}).items() if v.get("now") is not None}
    if use:
        best_s = max(use, key=lambda s: use[s]["now"])
        b = use[best_s]
        text = (f"Knowing the command it actually executed improves its next-model-step prediction by "
                f"{pct(b['now'], 2)} of the way to exact for {names[best_s]} (the median of the newest segment's "
                f"{b['n_now']:,} report windows, its best sense)")
        if b.get("hi") is not None:
            inside = b["now"] <= b["hi"]
            text += (f"; before the build, the middle 90 of every 100 windows read {pct(b['lo'], 2)} to {pct(b['hi'], 2)}"
                     + (". Inside that range after 344,400 steps, the addendum's section 20 treats it as evidence that "
                        "the action signal, its scaling, the architecture or the motor mapping is wrong."
                        if inside else ", and this reading is above that range."))
        else:
            text += "."
        bits.append(text)
    else:
        gains = [(name, SERIES[f"act.{sense}"]["get"](record)) for sense, name in SENSES] if record else []
        gains = [(n, v) for n, v in gains if v is not None]
        if gains:
            best = max(gains, key=lambda p: p[1])
            bits.append(f"In the newest report window, knowing the command it actually executed improves its "
                        f"next-model-step prediction by at most {pct(best[1], 2)} of the way to exact ({best[0]}); one "
                        f"window is too noisy to judge against the lineage (its spread is not loaded).")
        else:
            bits.append("The newest training record carries no prediction audit, so action use is unmeasured.")
    return dict(title=title, detail=" ".join(bits))


_STEP_RATE = re.compile(r"model steps per second ([\d.]+) under one third of own hour-1 baseline ([\d.]+)")


def stop_writer(stop):
    """Which program wrote a stop_reason.json, in words. Its 'writer' field names it (the motor fixes
    kill-line watcher writes one, the deploy branch's review, point 10); a stop without one is the
    hourly watcher's."""
    writer = str((stop or {}).get("writer") or "") if isinstance(stop, dict) else ""
    if "motor_fixes_watch" in writer:
        return ("the motor fixes kill-line watcher (docs/gain_of_function_v4/motor_fixes_watch.py, a program "
                "beside training that applies the kill lines of the eye-row convention and executed-fields "
                "deploy)")
    if "train_mimo" in writer:
        # the trainer's own stop after repeated physical faults of the worlds (the deploy branch's second
        # review, point 7: it was named the hourly watcher's)
        return "the trainer itself, after repeated physical faults of the worlds"
    if writer:
        shown = printable(writer)
        return ("the program " + shown) if shown else "the program stop_reason.json names"
    return "the hourly watcher program (a script beside training that reads the stats on a schedule)"


MOTOR_WATCH_STATUS = "motor_fixes_watch_status.json"   # in Saved/mimo_bridge (motor_fixes_watch.py)
MOTOR_WATCH_SILENT_POLLS = 5                            # the watcher's own --ensure line for "silent"


def _clock_words(t):
    return time.strftime("%I:%M %p", time.localtime(float(t))) if isinstance(t, (int, float)) else "an unrecorded time"


def motor_watch_note(saved_dir, now=None, run=None):
    """The motor fixes kill-line watcher's news for a dashboard, from the status file it writes at every
    poll (Saved/mimo_bridge/motor_fixes_watch_status.json; the deploy branch's second review, finding 6:
    a waiting stop had no reader but the watcher's own log), or None when there is none: a stop it found
    and waits to write, with what it waits for and for how long; a stop it wrote that no trainer has
    confirmed; a deploy relaunch that also changed the decoder's position rows or a limb flag; a watcher
    that has not polled for five of its polls; a watch that is complete (its updates judged, the stops
    written during it, and the eye blocks it could not judge) or was ended by hand (--end, at a revert),
    whose watcher has stopped on purpose; a relaunch that reverts the deploy's settings, which the watcher
    does not follow. run: a run folder name; the note is then given only when the watcher watches that
    run."""
    try:
        st = json.loads((Path(saved_dir) / MOTOR_WATCH_STATUS).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(st, dict):
        return None
    if run is not None and Path(str(run)).name not in (st.get("runs") or []):
        return None
    now = time.time() if now is None else float(now)
    who = "The motor fixes kill-line watcher"
    notes = []
    pending = st.get("pending_stop")
    if isinstance(pending, dict):
        since = pending.get("since")
        waited = ("for %d minutes" % max(0, round((now - float(since)) / 60.0))
                  if isinstance(since, (int, float)) else "since an unrecorded time")
        notes.append("%s found a crossing and has waited %s to stop training, because %s: %s"
                     % (who, waited, pending.get("waiting_for"), plain_stop_reason(pending.get("reason"))))
    open_stop = st.get("open_stop")
    if isinstance(open_stop, dict):
        running = sorted((open_stop.get("trainers_running") or {}).keys())
        notes.append("%s wrote a stop at %s that no trainer has confirmed yet%s: %s"
                     % (who, _clock_words(open_stop.get("time")),
                        "; trainers still run in %s" % ", ".join(running) if running else "",
                        plain_stop_reason(open_stop.get("reason"))))
    changed = st.get("same_relaunch_changes")
    if isinstance(changed, dict) and changed:
        notes.append("%s reports that the deploy relaunch also changed %s, which its switch-jump verdicts "
                     "do not cover" % (who, ", ".join("--%s" % str(k).replace("_", "-") for k in changed)))
    done = st.get("complete")
    if st.get("phase") == "complete" and isinstance(done, dict):
        # the deploy's one watch ended with no kill line crossed (the operator's decision of 2026-09-25: F4
        # and F1 together, one watch of 6,000 updates); the watcher then ends, so it is never "silent"
        span = [s for s in (done.get("steps") or []) if isinstance(s, (int, float))]
        blind = [int(s) for s in (done.get("eye_blind_blocks") or []) if isinstance(s, (int, float))]
        stops = [s for s in (done.get("stops") or []) if isinstance(s, dict)]
        notes.append("%s finished its watch%s %s%s"
                     % (who, (" of the %s updates after the switch (lineage steps %s to %s)"
                              % (format(int(span[1] - span[0]), ","), format(int(span[0]), ","),
                                 format(int(span[1]), ",")) if len(span) == 2 else ""),
                        ("with no kill line crossed" if not stops else
                         "after %d stop%s it wrote (%s), with no crossing after %s" % (
                             len(stops), "s" if len(stops) > 1 else "",
                             "; ".join(plain_stop_reason(s.get("reason")) for s in stops),
                             "them" if len(stops) > 1 else "it")),
                        ("; its eye-stop line could not judge the eye block%s that start%s at lineage step%s %s, "
                         "whose observer records never came" % ("s" if len(blind) > 1 else "",
                                                                "" if len(blind) > 1 else "s",
                                                                "s" if len(blind) > 1 else "",
                                                                ", ".join(format(s, ",") for s in blind))
                         if blind else "")))
    ended = st.get("ended")
    if st.get("phase") == "ended" and isinstance(ended, dict):
        notes.append("%s's watch was ended by hand at %s (%s)" % (who, _clock_words(ended.get("time")),
                                                                 printable(str(ended.get("reason") or ""))
                                                                 or "no reason written"))
    reverted = st.get("not_followed")
    if isinstance(reverted, dict) and reverted:
        notes.append("%s does not watch %s: %s the deploy's settings" % (
            who, ", ".join(sorted(str(k) for k in reverted)), "they revert" if len(reverted) > 1 else "it reverts"))
    t, poll = st.get("time"), st.get("poll_s")
    # a confirmed or released stop no longer ends the watch (the third review, finding A), so only a complete
    # or an ended watch's watcher has stopped on purpose
    if (st.get("phase") not in ("complete", "ended", None) and isinstance(t, (int, float))
            and isinstance(poll, (int, float)) and now - float(t) > MOTOR_WATCH_SILENT_POLLS * float(poll)):
        notes.append("%s last polled %d minutes ago (its process may have ended; run its --ensure command)"
                     % (who, round((now - float(t)) / 60.0)))
    return "; ".join(notes) + "." if notes else None


def plain_stop_reason(reason):
    """The watcher program's stop reason in plain words (known forms rewritten, others quoted)."""
    reason = str(reason or "")
    match = _STEP_RATE.search(reason)
    if match:
        speed, base = float(match[1]), float(match[2])
        return (f"the trainer ran {speed:.3f} model steps per second, under its stop line of one third of its "
                f"own first-hour speed of {base:.3f} steps per second (the line is {base / 3:.3f} steps per "
                f"second) on two hourly checks in a row")
    text = printable(reason)
    return f"\"{text}\"" if text else ("no reason was written" if not reason else
                                      "a reason written in a form this page does not print (read stop_reason.json)")
