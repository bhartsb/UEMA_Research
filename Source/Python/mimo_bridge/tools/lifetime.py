# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The whole life, on fixed yardsticks (operator 2026-09-03: "I really have zero
idea if our model is that much better from when we started").

Every run's stats survive on disk keyed by the lifetime step counter (checkpoint
lineage). This sweeps the runs the active model descends from (since 2026-09-22; before,
every run folder, 12 CPU-seconds per cycle) and answers, per yardstick, in absolute terms:
when it was first measurable, what it read then, its best, and what it reads now
- one plain sentence each, plus the raw series. Yardsticks were BORN on different
days (order-recognition did not exist before 2026-09-01; the mobile before
09-02); that is part of the story, not a gap.

Writes Saved/mimo_bridge/lifetime.json and lifetime.html. The dashboard links it
and rebuilds it when older than an hour, so the operator's refresh loop keeps it
current.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import math
from pathlib import Path
from html import escape

SAVED = r"C:\Users\bhart\Documents\UEMA_devA_private\Saved\mimo_bridge"
EXP = os.path.join(SAVED, "experiments")
OUT_JSON = os.path.join(SAVED, "lifetime.json")
OUT_HTML = os.path.join(SAVED, "lifetime.html")
# The code that turns a record into this page's cached values (YARDSTICKS and _extract) is this file;
# its tag joins the cache signature, so an edit never serves values the old code extracted and two
# code versions keep separate caches (fix-readout-dashboards 2026-09-23; lineage_series.read_rows
# adds its own reader's tag). Taken at import: it names the code this process runs.
try:
    SOURCE_TAG = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:16]
except OSError:
    SOURCE_TAG = ""

# yardstick -> (extractor(rec) -> float|None, lower_is_better, plain name, unit note)
def _g(r, *path):
    cur = r
    for k in path:
        cur = (cur or {}).get(k)
    return cur


YARDSTICKS = {
    "vision_next": (lambda r: _g(r, "next_pe", "vision"), True,
                    "Predicting the next instant of VISION", "error; smaller = better"),
    "tactile_next": (lambda r: _g(r, "next_pe", "tactile"), True,
                     "Predicting the next instant of TACTILE", "error; smaller = better"),
    "proprio_next": (lambda r: _g(r, "next_pe", "proprioception"), True,
                     "Predicting the next instant of PROPRIOCEPTION", "error; smaller = better"),
    # obedience judged on movement BOUTS where they exist (rest-era runs, 09-07 on):
    # during pose-hold rests the raw held cosine goes negative BY CONSTRUCTION (the
    # hold command opposes drift) and read -0.18..-0.24 on a healthy run
    "follow": (lambda r: (r.get("follow_bout") if r.get("follow_bout") is not None
                          else r.get("cmd_response_cos_held")), False,
               "The body moves the way it was told",
               "direction agreement -1..1: follow_bout = one 10 ms frame per model step in movement bouts "
               "(not a command's 250 ms life); the held command's 250 ms reading only on runs without bouts"),
    "eyes": (lambda r: r.get("task_hand_fov"), False,
             "Eyes on its own hand", "fraction of frames; not a skill score"),
    "imag": (lambda r: r.get("imagination_loss"), True,
             "Imagining a few steps ahead", "error; smaller = better"),
    "rest": (lambda r: r.get("rest_over_bout_speed"), True,
             "Rest is real (rest speed vs bout speed)", "1.0 = no rest; gate 0.5"),
    "rank1": (lambda r: _g(r, "tenure_tok", "rank1"), False,
              "Picking its OWN command's effect from the lineup", "fraction right; chance ~0.067"),
    "margin": (lambda r: (100.0 * _g(r, "tenure_gap", "pose") / _g(r, "tenure_pe", "pose"))
               if _g(r, "tenure_pe", "pose") else None, False,
               "Command-aware prediction beats do-nothing by", "percent; the hinge demands +4"),
    "vis_gate": (lambda r: _g(r, "tenure_tok", "vis_gate_open"), False,
                 "Share of lessons vision is allowed to learn from", "fraction of command endings"),
    "vis_adv": (lambda r: _g(r, "agency", "vis_adv"), False,
                "Vision's own 'I did that' evidence", "advantage over decoys"),
    "strikes": (lambda r: r.get("task_mobile_hand"), False,
                "Hand strikes on the mobile", "fraction of steps"),
    "variance": (lambda r: _g(r, "token_temporal_std", "vision"), False,
                 "Vision's expressiveness (collapse guard)", "temporal spread; <0.15 = collapse"),
}


def _gof(key):
    from mimo_bridge.tools.lineage_series import SERIES
    return SERIES[key]["get"]


def _best_action_use(r):
    from mimo_bridge.tools.lineage_series import SERIES
    vals = [SERIES[f"act.{m}"]["get"](r) for m in ("vision", "tactile", "proprioception")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# The gain-of-function yardsticks (addendum section 19, items 1-6 and 10; items 7-9 are probe
# results, listed from the probe index). Definitions, scale ends, display forms and goal lines live
# in tools/lineage_series.py (GOF_SERIES names the series each yardstick reads), so this page and the
# development dashboard read one meaning and print it the same way (DESIGN.md section 11).
GOF_SERIES = {"gof_act": "act.vision", "gof_cc": "cc.chosen_best", "gof_ctl": "ctl.agreement",
              "gof_follow": "follow", "gof_lp": "lp.value", "gof_copy": "pers.proprioception",
              "gof_rank": "wr.lead", "gof_syn": "syn.following", "gof_touch": "ccx.onset_self",
              "gof_touch_paired": "ccx.paired", "gof_vest": "vest.gain", "gof_posture": "posture.max"}
GOF_YARDSTICKS = {
    "gof_act": (_best_action_use, False, "1. Knowing its own executed command improves prediction (best sense)",
                "the control is another body's command"),
    "gof_cc": (lambda r: _gof("cc.chosen_best")(r), False,
               "1. Chose the plan it predicted would feel best", "needs --causal-chain on; chance is 1 in (1 + alternatives)"),
    "gof_ctl": (lambda r: _gof("ctl.agreement")(r), False, "2. What the body did matched what it expected",
                "needs --controllability on; the control is the same body's result against another body's command"),
    "gof_follow": (lambda r: _gof("follow")(r), False, "2. The body turns the way it was commanded",
                   "logged on every segment"),
    "gof_lp": (lambda r: _gof("lp.value")(r), False, "3. Learning progress: prediction errors falling",
               "needs --learning-progress on; frozen weights read -7.1% to -1.6% with no learning"),
    "gof_copy": (lambda r: _gof("pers.proprioception")(r), False,
                 "3. Proprioception predicted better than a copy of the present", "logged on every segment"),
    "gof_rank": (lambda r: _gof("wr.lead")(r), False, "4. The leading channel's share of the winning goal's lead",
                 "needs --wellness-rank normalized; read against its share of the weights"),
    "gof_syn": (lambda r: _gof("syn.following")(r), False, "5. The body follows commands written through the synergy map",
                "needs --motor-synergy on"),
    "gof_touch": (lambda r: _gof("ccx.onset_self")(r), False, "6. Own-body touch foreseen before it begins",
                  "needs --contact-head on; the control is a base-rate guess on the same patches and model steps"),
    "gof_touch_paired": (lambda r: _gof("ccx.paired")(r), False, "6. Self-touch onsets foreseen with its own command",
                         "needs --contact-head on; the control is another body's command on the same onsets"),
    "gof_vest": (lambda r: _gof("vest.gain")(r), False, "10. Vestibular predicted better than \"nothing changes\"",
                 "needs --vestibular-sense on"),
    "gof_posture": (lambda r: _gof("posture.max")(r), False, "10. Most-turned body, degrees from lying on its back",
                    "90 degrees = on its side"),
}
YARDSTICKS.update(GOF_YARDSTICKS)
# The control each gain-of-function yardstick must beat, read from the same runs and printed beside
# it (reviews/dashboards.md B1: a reading shown without its control's value reads as a share of the
# way from knowing nothing; verify-fix-dashboards 2026-09-23). {yardstick: (control key, series)}
GOF_CONTROLS = {"gof_cc": ("gof_cc_chance", "cc.chance"), "gof_ctl": ("gof_ctl_control", "ctl.control"),
                "gof_lp": ("gof_lp_control", "lp.control"),
                "gof_touch": ("gof_touch_control", "ccx.onset_self_control"),
                "gof_touch_paired": ("gof_touch_paired_control", "ccx.success_control")}
for _key, (_ctl, _series) in GOF_CONTROLS.items():
    YARDSTICKS[_ctl] = ((lambda s: lambda r: _gof(s)(r))(_series), False, "control of " + _key, "")


def _extract(r):
    out = []
    for fn, _lb, _nm, _un in YARDSTICKS.values():
        try:
            v = fn(r)
        except (TypeError, ValueError, ZeroDivisionError, KeyError, AttributeError):
            v = None
        out.append(v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None)
    return out


def sweep(bounds=None):
    """One entry per run on the active lineage: the median of each yardstick over the last
    quarter of that run's kept records. Only the lineage's runs are read (the bounds), and
    each stats file is read from where the previous sweep stopped (lineage_series.read_rows):
    sweeping all 248 run folders cost 12 CPU-seconds every cycle (measured 2026-09-22)."""
    from mimo_bridge.tools.lineage_series import read_rows
    runs = []
    bounds = bounds or {}
    keys = list(YARDSTICKS)
    sig = "lifetime_v2:" + ",".join(keys) + ":" + SOURCE_TAG
    cache = os.path.join(SAVED, "lineage_series", "lifetime")
    for d, limit in bounds.items():
        p = os.path.join(d, "replay_training_stats.jsonl")
        if not os.path.isfile(p):
            continue
        rows, meta, _st = read_rows(p, cache, extractor=_extract, sig=sig)
        mi = [row for row in rows if limit is None or row[0] <= limit]
        if len(mi) < 8:
            continue
        tail = mi[max(0, 3 * len(mi) // 4):]
        # the code the run launched on (its launch record), so a gauge whose definition changed at a
        # commit can be marked at the first run that ran it (lineage_series.seam_text)
        launch = next((m for m in (meta or {}).get("launches", [])
                       if m.get("code_commit") and (limit is None or (m.get("step") or 0) <= limit)), None)
        entry = {"run": os.path.basename(d),
                 "linked": True,
                 "step": int(mi[-1][0]),
                 "first_step": int(mi[0][0]),
                 "code_commit": launch.get("code_commit") if launch else None,
                 "ts": float(mi[-1][1]),
                 "when": time.strftime("%Y-%m-%d", time.localtime(mi[-1][1]))}
        for col, key in enumerate(keys):
            vals = sorted(v for v in (row[2][col] for row in tail) if v is not None)
            entry[key] = (vals[len(vals) // 2] if vals else None)   # median of the tail
        st = os.path.join(d, "situation_test.json")
        if os.path.isfile(st):
            try:
                entry["situation"] = json.load(open(st)).get("established")
            except (OSError, ValueError):
                pass
        runs.append(entry)
    # sort by TIME, not step (external review 2026-09-08 finding 7: a reverted
    # branch with a higher step count was shadowing the later mainline as "latest")
    runs.sort(key=lambda e: e["ts"])
    return runs


def _fmt_step(x):
    return (f"{x/1e6:.2f}M" if x >= 1e6
            else f"{x/1e3:.0f}K" if x >= 1000 else f"{x:.0f}")


# the story's dated turning points, drawn on every chart whose data spans them
ERAS = [("2026-09-02", "mobile"), ("2026-09-04", "batting"),
        ("2026-09-07", "rest"), ("2026-09-08", "eyes-pull")]

# goal/reference line per yardstick: (value, label) - drawn dashed on the chart.
# A floor is labeled a floor, never a goal; an unmeasured goal says so.
GOAL_LINES = {
    "eyes": (0.03, "historical 3% alert; not a skill criterion"),
    "vis_adv": (0.011, "the measured noise; real only above it"),
    "variance": (0.15, "collapse under 15 in 100"),
    "margin": (4.0, "lesson demands +4%"),
    "rank1": (0.067, "chance floor, 7 in 100 (floor, not a goal)"),
    "rest": (0.5, "gate: rest at half the active speed"),
    "imag": (None, "goal line unmeasured"),
    "strikes": (None, "goal line unmeasured"),
}


def _goal_for(key):
    """(value in the display form, label) for a gain-of-function yardstick's goal line, from its
    series in lineage_series (the reachable-posture following measured on physics alone; the
    planner's choice share has none: it does not aim at predicted wellness)."""
    from mimo_bridge.tools.lineage_series import SERIES, say_ref, shown
    spec = SERIES[GOF_SERIES[key]]
    if not spec.get("goal"):
        return (None, "goal line unmeasured")
    g = spec["goal"][0]
    return (shown(spec["display"], g), f"goal line {say_ref(spec, g)}: a posture the body really reached, on physics alone")


def _spark(xs, ys, w=560, h=86, log_y=False, goal=None, fmt=None):
    """The chart carries the story (operator 2026-09-08, three passes): a log axis
    for error curves (linear crushed an 8x gain flat), born -> today with the
    multiple, the GOAL LINE dashed so distance-to-goal is visible, and the dated
    turning points (mobile, batting, rest, eyes-pull) as vertical marks. X axis
    is calendar time - in steps, the decisive last week was an invisible sliver."""
    if len(ys) < 2:
        return ""
    import math
    x0, x1 = min(xs), max(xs)
    born, now = ys[0], ys[-1]
    gval, glabel = goal if goal else (None, None)
    tfm = ((lambda y: math.log10(max(y, (max(min(v for v in ys if v > 0), 1e-9) * 0.5
                                        if any(v > 0 for v in ys) else 1e-9))))
           if log_y else (lambda y: y))
    ty = [tfm(y) for y in ys]
    lo, hi = min(ty), max(ty)
    if gval is not None:
        lo, hi = min(lo, tfm(gval)), max(hi, tfm(gval))
    span_x = max(x1 - x0, 1)
    span_y = max(hi - lo, 1e-12)
    # STRICT LABEL BANDS (operator 2026-09-08: text sat on text) - header y<=12,
    # plot 16..16+ph with the goal label clamped INSIDE it, two era-label rows,
    # then the date row. No two bands share y-space.
    h = 100
    top = 16
    ph = h - top - 34          # 34 = era rows (2x10) + date row (14)
    def Y(v):
        return top + ph - (v - lo) / span_y * ph
    def X(x):
        return 4 + (x - x0) / span_x * (w - 8)
    pts = " ".join(f"{X(x):.1f},{Y(t):.1f}" for x, t in zip(xs, ty))
    parts = [f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;height:{h}px">']
    # era marks first (behind the data); labels alternate two rows of their own band
    for di, (ds, dl) in enumerate(ERAS):
        ts = time.mktime(time.strptime(ds, "%Y-%m-%d"))
        if x0 <= ts <= x1:
            ex = X(ts)
            ex_lab = min(max(ex, 34.0), w - 42.0)   # keep the label inside the frame
            parts.append(f'<line x1="{ex:.1f}" y1="{top}" x2="{ex:.1f}" y2="{top+ph}" '
                         f'stroke="#c7cdd4" stroke-width="1"/>'
                         f'<text x="{ex_lab:.1f}" y="{h - 24 + (di % 2) * 10}" font-size="9" '
                         f'fill="#8a949e" text-anchor="middle">{dl}</text>')
    if gval is not None:
        gy = Y(tfm(gval))
        # label goes below the line when the line is high, above when low - and is
        # clamped into the plot band either way
        gy_lab = gy + 11 if gy < top + ph / 2 else gy - 4
        gy_lab = min(max(gy_lab, top + 9), top + ph - 2)
        parts.append(f'<line x1="4" y1="{gy:.1f}" x2="{w-4}" y2="{gy:.1f}" '
                     f'stroke="#b8791a" stroke-width="1" stroke-dasharray="5,4"/>'
                     f'<text x="{w-6}" y="{gy_lab:.1f}" font-size="9.5" fill="#b8791a" '
                     f'text-anchor="end">{escape(glabel)}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.6"/>')
    parts.append(f'<circle cx="{X(xs[-1]):.1f}" cy="{Y(ty[-1]):.1f}" r="2.6" fill="#0a7"/>')
    tag = " &middot; log axis" if log_y else ""
    note = f" &middot; {escape(glabel)}" if (gval is None and glabel) else ""
    if log_y:
        # an error in encoded units has no lived anchor: only how many times it changed is printed
        # (DESIGN.md section 11)
        change = ("the latest window's error equals the first window's" if not (born and now and born != now)
                  else f"the latest window's error is {born / now:.1f} times lower than the first window's"
                  if born > now else f"the latest window's error is {now / born:.1f} times the first window's")
        parts.append(f'<text x="4" y="10" font-size="9.5" fill="#66727c">'
                     f'<tspan fill="#0a7" font-weight="700">{change}</tspan>{tag}{note}</text>')
    else:
        from mimo_bridge.tools.plain_numbers import fixed as _fx
        fmt = fmt or (lambda v: _fx(v, 3))
        parts.append(f'<text x="4" y="10" font-size="9.5" fill="#66727c">first window {escape(fmt(born))} &rarr; '
                     f'latest window <tspan fill="#0a7" font-weight="700">{escape(fmt(now))}</tspan>{tag}{note}</text>')
    d0 = time.strftime("%b %d", time.localtime(x0))
    d1 = time.strftime("%b %d", time.localtime(x1))
    parts.append(f'<text x="4" y="{h-3}" font-size="9.5" fill="#66727c">{d0} (first measured)</text>')
    parts.append(f'<text x="{w-4}" y="{h-3}" font-size="9.5" fill="#66727c" '
                 f'text-anchor="end">{d1} (latest window)</text></svg>')
    return "".join(parts)


def _spark_for(runs, key):
    xs = [e["ts"] for e in runs if e.get(key) is not None and e.get("ts")]
    ys = [e[key] for e in runs if e.get(key) is not None and e.get("ts")]
    return _spark(xs, ys, log_y=bool(YARDSTICKS[key][1]), goal=GOAL_LINES.get(key))


def _latest(runs, key):
    vals = [(e["step"], e[key]) for e in runs if e.get(key) is not None]
    return vals[-1][1] if vals else None


def _first(runs, key):
    vals = [(e["step"], e[key]) for e in runs if e.get(key) is not None]
    return vals[0][1] if vals else None


def _series(runs, key):
    return ([e["step"] for e in runs if e.get(key) is not None],
            [e[key] for e in runs if e.get(key) is not None])


def build(max_age_s=1800, force=False):
    """Rebuild diagnostic history, separating the active ancestry from other runs."""
    from mimo_bridge.tools.development_dashboard import training_state, ancestry_bounds, atomic_write
    # the marker changes whenever the page's content changes, so a restarted updater rebuilds the
    # page at once instead of serving the old code's copy for up to max_age_s (reviews/dashboards.md
    # m1); v3 = the page prints no bare score decimals (DESIGN.md section 11); v4 = each
    # gain-of-function yardstick prints its control's value beside it; v5 = item 5's yardstick reads
    # only following split by who wrote the order (every graded order is a stand-in); v6 = the
    # following card names follow_bout's 10 ms horizon (fix-rows-r2-dashboard.py); v7 = the following
    # cards say where the following gauge changed (commit b55ee9a4d; fix-rows-r2-lineage_series)
    marker = "lineage-history-v8"
    target = Path(OUT_HTML)
    if (not force and target.exists() and time.time()-target.stat().st_mtime < max_age_s
            and marker in target.read_text(encoding="utf-8")):
        return str(target)
    root = Path(SAVED).parents[1]
    training = training_state(root, time.time())
    runs = sweep(ancestry_bounds(root, training))
    atomic_write(Path(OUT_JSON), json.dumps(runs, indent=2, allow_nan=False))
    linked = [r for r in runs if r["linked"]]
    other = [r for r in runs if not r["linked"]]
    archive = target.with_name("lifetime_before_review_2026-09-12.html")
    if target.exists() and not archive.exists() and marker not in target.read_text(encoding="utf-8"):
        archive.write_bytes(target.read_bytes())
    from mimo_bridge.tools.lineage_series import SERIES, goal_text as _goal_text, say, scale_text, shown
    from mimo_bridge.tools.lineage_series import seam_text
    from mimo_bridge.tools.plain_numbers import fixed, pct
    # the linked runs as segments, oldest first, for the sentence that says where a gauge changed
    seam_runs = [dict(code_commit=r.get("code_commit"), label=r["run"], first_step=r.get("first_step"))
                 for r in linked]

    def seam_html(series_key):
        text, _seg = seam_text(SERIES[series_key], seam_runs)
        return f"<p>{escape(text)}</p>" if text else ""

    def before_seam(spec, observed):
        """The newest reading is on the gauge before the series' seam (no run runs the change yet, or
        the newest run is earlier), while the goal line was measured after it: no distance is given."""
        text, seg = seam_text(spec, seam_runs)
        if not text or not observed:
            return False
        order = [r["run"] for r in linked]
        return seg is None or order.index(observed[-1]["run"]) < order.index(seg["label"])

    def number(value, percent=False):
        return "unavailable" if value is None else (f"{100*value:.1f}%" if percent else fixed(value, 3))
    def table(entries):
        rows = []
        for r in reversed(entries):
            rows.append("<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in (
                r["when"], r["run"], f'{r["step"]:,}', number(r["eyes"],True),
                number(r["strikes"],True), "unavailable" if r["rest"] is None else pct(r["rest"], 2)))+"</tr>")
        return ('<div class="scroll"><table><thead><tr><th>Date</th><th>Training run</th>'
                '<th>Training step</th><th>Hand in view</th><th>Hand touches mobile</th>'
                '<th>Rest speed as a share of active speed</th></tr></thead><tbody>'+"".join(rows)+"</tbody></table></div>")
    # the senses this page names, each in the model's own word with its plain meaning
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    sense_words = ("Sense names are the model's own: "
                   + "; ".join(f"{m} is {SENSE_DEFINITIONS[m]}" for m in ("proprioception", "vestibular")) + ".")
    cards = []
    # (key, title, note, display: value -> (the number in words, the value plotted, its axis unit))
    follow_words = lambda v: (f"{fixed(100 * v, 2)}% of the way from unrelated to exact", 100 * v)   # noqa: E731
    definitions = [
        ("eyes", "Seeing its own hand", "Share of recorded frames with a hand in view. This does not measure recognizing it as its own.",
         lambda v: (pct(v, 2) + " of frames", 100 * v)),
        ("strikes", "Touching the mobile", "Share of recorded steps with hand contact. Repeated steps in one contact are not separate successful reaches.",
         lambda v: (pct(v, 2) + " of steps", 100 * v)),
        ("rest", "Movement during rest", "Speed during rest as a share of speed during active movement: 0% means no rest movement, 100% means equal speeds. This does not establish purposeful rest.",
         lambda v: (pct(v, 2) + " of the active speed", 100 * v)),
        ("follow", "Following a command", "Direction agreement as a share of the way from 'unrelated' (0%) to 'the same direction' (100%); below 0% = against the command. On runs with movement bouts it is read on one 10 ms physics frame at the end of each 50 ms model step, in bouts only, not over a command's whole 250 ms; runs without bouts show the whole command's 250 ms reading instead, so the line changes what it measures where the bouts began.",
         follow_words),
    ]
    for key, title, note, disp in definitions:
        observed = [r for r in linked if r.get(key) is not None]
        value = observed[-1][key] if observed else None
        chart = _spark([r["ts"] for r in observed], [disp(r[key])[1] for r in observed],
                       fmt=lambda v: fixed(v, 2) + "%")
        date = observed[-1]["when"] if observed else "no linked measurement"
        # the following card reads follow_bout, whose gauge changed at commit b55ee9a4d
        seam = seam_html("follow_bout") if key == "follow" else ""
        cards.append(f'<article><h2>{title}</h2><strong>{escape("unavailable" if value is None else disp(value)[0])}</strong>'
                     f'<p>Latest available window: {date}</p><p>{escape(note)}</p>{seam}{chart}</article>')
    gof_cards = []
    for key, (_fn, _lb, title, note) in GOF_YARDSTICKS.items():
        spec = SERIES[GOF_SERIES[key]]
        observed = [r for r in linked if r.get(key) is not None]
        value = observed[-1][key] if observed else None
        goal = _goal_for(key)
        unit_s = spec["display"]["unit"]
        chart = (_spark([r["ts"] for r in observed], [shown(spec["display"], r[key]) for r in observed], goal=goal,
                        fmt=lambda v: fixed(v, 2) + unit_s) if len(observed) > 1 else "")
        where = (f"Latest linked run: {observed[-1]['run']} ({observed[-1]['when']})" if observed else
                 "Unmeasured: no run on this lineage carries it (its flag is off, or the field is newer than the runs)")
        control = ""
        if value is not None and key in GOF_CONTROLS:
            ctl_key, ctl_series = GOF_CONTROLS[key]
            ctl_value = observed[-1].get(ctl_key)
            control = (f"Its control in the same run: {SERIES[ctl_series]['title'][:1].lower()}"
                       f"{SERIES[ctl_series]['title'][1:]}, {say(SERIES[ctl_series], ctl_value, full=False)}. "
                       if ctl_value is not None else "Its control is not in the same run's records. ")
        gof_cards.append(f'<article><h2>{escape(title)}</h2><strong>{escape("unmeasured" if value is None else say(spec, value))}</strong>'
                         f'<p>{escape(control)}{escape(scale_text(spec))} {escape(note[:1].upper() + note[1:])}. '
                         f'{escape(_goal_text(spec, value, before_seam=before_seam(spec, observed)))}</p>'
                         f'<p>{escape(where)}</p>{seam_html(GOF_SERIES[key])}{chart}</article>')
    try:
        from mimo_bridge.tools.lineage_series import probe_results
        probes = probe_results(root, ancestry_bounds(root, training)).get("items", {})
    except (OSError, ValueError, TypeError):
        probes = {}
    for item, name in (("7", "7. Gaze probes"), ("8", "8. Body-control probes"),
                       ("9", "9. Arm into view and self-touch probes"), ("10", "10. Posture and balance probes")):
        entries = [e for e in probes.get(item, []) if e.get("latest")]
        if entries:
            body = "".join(f'<p>{escape(e["words"]["sentence"])} {escape(e["words"]["goal"])}</p>'
                           for e in entries if e.get("words"))
        else:
            body = '<p>Unmeasured: no probe result on this lineage in Saved/mimo_bridge/gof_probes/index.jsonl.</p>'
        gof_cards.append(f'<article><h2>{name}</h2>{body}</article>')
    archive_link = ('<a href="lifetime_before_review_2026-09-12.html">Archived earlier page (contains withdrawn achievement claims)</a>'
                    if archive.exists() else "")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>UEMA training history</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f6f8f7;color:#162a32;font:16px/1.6 'Segoe UI',sans-serif}}
main{{max-width:1200px;padding:32px 24px;margin:auto}}h1{{font-size:38px;line-height:1.15}}h2{{font-size:20px}}a{{color:#12645e}}
.notice{{border-left:4px solid #895212;padding:20px;background:#fbefd9}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:28px 0}}
article{{padding:24px;background:white;border:1px solid #d8e0e2}}article strong{{font-size:30px}}article p{{font-size:14px;color:#536770}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d8e0e2}}
details{{margin:24px 0}}summary{{cursor:pointer;color:#12645e}}footer{{font-size:13px;margin-top:30px}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}h1{{font-size:30px}}}}
</style></head><body><main data-version="{marker}">
<nav><a href="dashboard.html#history">Development dashboard</a> &middot; <a href="dashboard_legacy.html">Old diagnostic dashboard</a></nav>
<h1>What has changed over training?</h1>
<p class="notice"><b>Intentional rolling and purposeful recovery remain unproved.</b> Training statistics cannot establish those abilities. Use the development dashboard for dated behavior tests.</p>
<p>Current run: <b>{escape(training.get('run_name') or 'unknown')}</b>. {len(linked)} runs with enough measurements are linked to its saved ancestry.</p>
<p>Each point summarizes the last quarter of the available records in one run. Earlier runs stop at the checkpoint that was carried forward. Different worlds, sensors and controls can change these numbers; these are not controlled before-and-after skill comparisons.</p>
<div class="cards">{''.join(cards)}</div>
<h2>Gain-of-function yardsticks (addendum items 1-10)</h2>
<p>Each item's own measurement is written only while its flag is on; "unmeasured" means no run on this lineage carries it, never zero. The development dashboard's Over-time tab charts every window of the lineage.</p>
<p>{escape(sense_words)}</p>
<div class="cards">{''.join(gof_cards)}</div>
<h2>Records carried into this model</h2>{table(linked) if linked else '<p>No linked run has enough records yet.</p>'}
<p>Only the runs this model descends from are read. Sweeping every run folder cost 12 CPU-seconds every cycle (measured 2026-09-22), so other runs are no longer listed here; their files are unchanged in their folders.</p>
<footer>Built {time.strftime('%Y-%m-%d %H:%M:%S')}. The updater rebuilds this history every 30 minutes; this page checks for an updated copy every minute.<br><a href="lifetime.json">Raw history</a> &middot; {archive_link}</footer>
</main></body></html>"""
    atomic_write(target, page)
    return str(target)


if __name__ == "__main__":
    print(build(force=True))
