# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Deaf senses: which bodies' tokens of a sense have stopped changing (2026-09-25).

THE READING. Every stats record carries the per-body collapse watch (`token_temporal_std_bodies`,
`lem_model.LemLite.token_temporal_std_bodies`): per sense, one number per body, the standard deviation
over time of that body's raw token (each of its numbers over the collapse watch's 200 samples, one every
5th model step, about 50 s), averaged over the numbers. After the registry's output LayerNorm (no gain,
no bias) a token has root mean square 1 per number, so the reading runs from 0 (the same token every
moment) to at most 1 (an unrelated token every moment); pages print it as a percentage of that range.

A body is DEAF in a sense when its reading stays under that sense's line: its token of that sense is
almost the same every moment. Two causes give that, and the reading alone cannot tell them apart: the
encoder stopped answering that body (the defect found on H84,
docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md), or the body's input held still (a body
lying still, a view that does not change; on 2026-09-14 H81's touch watch read 5.3% on a settled posture
pressing the same contacts while a probe read the same encoder at 21-32%). For body position,
`--proprio-token-repair within_body` records the ratio that separates the two (the token's spread per
posture spread).

THE LINES, from 480 body-windows per sense: the replay rings of 10 H84 checkpoints (Sept 18 to Sept 25)
x 8 bodies x 6 windows of 50 s (docs/gain_of_function_v3/measurements/proprio_token_bodies_2026-09-25.json;
tests/test_deaf_senses.py recomputes every number below from that file). The hourly watcher judges each
body's mean over an hour of readings, so every line is also checked against each body's six-window
(300 s) mean, the nearest measured stand-in for an hour's mean:
- vision 20%: two groups. Single windows: 35 under 15%, every one from the one body lying face down;
  433 at 25% or more; 12 between. Six-window means: that body 6.9-19.8% at the 7 face-down checkpoints
  after its first, every other body 28.3-50.2%, none between.
- body position 20%: two groups. Single windows: 108 under 15%, 319 at 30% or more, 53 between.
  Six-window means: 18 at 4.0-13.7%, 62 at 22.1-51.1%, none between. (The 64-s readings of all 227 H84
  checkpoints give the same median count of deaf bodies, 2 before the Sept 23 deploy and 5 after, for
  any line from 18% to 25%.)
- touch 9%, hearing 29%, motor 39%: one group, no deaf body in any window. The line is the lowest single
  window (9.2%, 29.9%, 39.2%) rounded down: under it, a body's token changed less over the hour than any
  body's did in any 50 s of the 480.
- balance 1.6%: no live reading exists on H84 (every body 0% in all 96 windows since the Sept 23
  deploy). 1.6% is the lowest the placeholder channel before it ever read (384 windows), the same number
  the vestibular repair declared as its "still dead" kill line.
The placeholder channel (`intermind`) is not a sense and is never judged. A sense with no line here is
printed with "no line measured" and never named.

Bodies are numbered from 1 (the record's list index + 1), as the dashboard numbers them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

VERSION = "deaf_senses_v1"
MEASUREMENT = "docs/gain_of_function_v3/measurements/proprio_token_bodies_2026-09-25.json"

# THE SENSES GO BY THE MODEL REGISTRY'S NAMES in everything this module prints (the style guide "ONE WORD PER THING"; until
# 2026-10-03 it printed "touch", "hearing", "body position" and "balance"). A text that is read alone (the watcher's
# alert) gives each sense's plain meaning where it first names it; the meanings are the opening words of
# mimo_bridge/explainer/schema.py SENSE_DEFINITIONS, and tests/test_deaf_senses.py holds the two equal. On the
# Development page the glossary defines each name where it stands.
SENSE_WORDS = {s: s for s in ("vision", "tactile", "audio", "proprioception", "motor", "vestibular")}
SENSE_MEANS = {"vision": "what the two eyes see", "tactile": "the skin's sense of touch",
               "audio": "what the two ears hear",
               "proprioception": "where each tracked body part is, how it is turned and how fast it moves",
               "motor": "the command the body was last given, beside the pose each part reached",
               "vestibular": "the body's load and balance in 29 numbers"}
NOT_JUDGED = {"intermind": "the placeholder channel, not a sense"}
# the order senses are printed in
ORDER = ("proprioception", "vestibular", "vision", "tactile", "audio", "motor")

LINES = {"vision": 0.20, "proprioception": 0.20, "tactile": 0.09, "audio": 0.29, "motor": 0.39,
         "vestibular": 0.016}
REASONS = {
    "vision": ("the gap between the one body lying face down (6.9-19.8% over 300 s) and every other body "
               "(28.3-50.2%), 480 fifty-second windows on H84, Sept 18-25"),
    "proprioception": ("the gap between deaf bodies (4.0-13.7% over 300 s) and live ones (22.1-51.1%), 480 "
                       "fifty-second windows on H84, Sept 18-25"),
    "tactile": ("no deaf body measured: under it the token changed less than in any of 480 fifty-second "
                "windows on H84, Sept 18-25 (lowest 9.2%)"),
    "audio": ("no deaf body measured: under it the token changed less than in any of 480 fifty-second "
              "windows on H84, Sept 18-25 (lowest 29.9%)"),
    "motor": ("no deaf body measured: under it the token changed less than in any of 480 fifty-second "
              "windows on H84, Sept 18-25 (lowest 39.2%)"),
    "vestibular": ("the lowest the placeholder channel before it ever read (384 fifty-second windows on H84), "
                   "the vestibular repair's still-dead line; no live vestibular reading existed to measure (every "
                   "body 0% from the Sept 23 deploy to Sept 25)"),
}
SCALE = ("0% = the same token on every sampled model step, 100% = the most a token can change (an unrelated token "
         "on every sampled model step)")
CAUSES = ("the sense's encoder stopped answering that body, or the body's input held still (a body lying "
          "still, a view that does not change); this reading alone cannot tell which")


def finite(value) -> Optional[float]:
    """float(value) when it is a finite real number (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def pct(value) -> str:
    """A reading (0..1) as a percentage: whole under 100 from 10%, one decimal from 1% to 10%, two
    significant digits under 1%, "0%" for 0."""
    v = finite(value)
    if v is None:
        return "unmeasured"
    p = 100.0 * v
    if p == 0 or abs(p) >= 10:
        return "%.0f%%" % p
    if abs(p) >= 1:
        return "%.1f%%" % p
    return "%.2g%%" % p


def line_pct(line: float) -> str:
    """A line as the percentage it was declared at: 0.09 -> "9%", 0.016 -> "1.6%"."""
    return "%g%%" % round(100.0 * float(line), 1)


def word(sense: str) -> str:
    return SENSE_WORDS.get(sense, str(sense).replace("_", " "))


def with_meaning(sense: str) -> str:
    """A sense's name with its plain meaning, for a text that is read alone."""
    meaning = SENSE_MEANS.get(sense)
    return word(sense) if not meaning else "%s (%s)" % (word(sense), meaning)


def ordered(senses) -> List[str]:
    known = [s for s in ORDER if s in senses]
    return known + sorted(s for s in senses if s not in ORDER)


def judge(readings: Dict[str, Sequence]) -> List[dict]:
    """Every judged sense of `readings` ({sense: [reading per body]}), in print order: dict(sense, word,
    line, reason, bodies=[(body number from 1, reading)] under the line, measured=bodies with a reading).
    The placeholder channel and senses without a line are left out; a body without a finite reading is
    neither named nor counted."""
    out = []
    for sense in ordered(readings or {}):
        if sense in NOT_JUDGED or sense not in LINES:
            continue
        values = readings.get(sense)
        if not isinstance(values, (list, tuple)):
            continue
        measured = [(i + 1, finite(v)) for i, v in enumerate(values)]
        measured = [(n, v) for n, v in measured if v is not None]
        line = LINES[sense]
        out.append(dict(sense=sense, word=word(sense), line=line, reason=REASONS[sense],
                        bodies=[(n, v) for n, v in measured if v < line], measured=len(measured)))
    return out


def named(judged: List[dict]) -> List[dict]:
    """The judged senses with at least one body under the line."""
    return [j for j in judged if j["bodies"]]


def bodies_text(bodies, measured: int) -> str:
    """'all 8 bodies 0%' when every measured body is named with one printed value, else
    'body 1 9.1%, body 2 9.6%'."""
    shown = {pct(v) for _n, v in bodies}
    if measured > 1 and len(bodies) == measured and len(shown) == 1:
        return "all %d bodies %s" % (measured, shown.pop())
    return ", ".join("body %d %s" % (n, pct(v)) for n, v in bodies)


def which_bodies(bodies, measured: int) -> str:
    """'all 8 bodies', 'body 3', 'bodies 1, 2, 3, 6 and 8'."""
    numbers = [str(n) for n, _v in bodies]
    if measured > 1 and len(numbers) == measured:
        return "all %d bodies" % measured
    if len(numbers) == 1:
        return "body " + numbers[0]
    return "bodies " + ", ".join(numbers[:-1]) + " and " + numbers[-1]


def summary_text(readings: Dict[str, Sequence]) -> str:
    """Every sense's lowest and highest body reading with its line, in print order."""
    parts = []
    for sense in ordered(readings or {}):
        values = readings.get(sense)
        vals = [finite(v) for v in values] if isinstance(values, (list, tuple)) else []
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        low, high = pct(min(vals)), pct(max(vals))
        span = low if low == high else "%s to %s" % (low, high)
        if sense in NOT_JUDGED:
            tail = "%s, not judged" % NOT_JUDGED[sense]
        elif sense in LINES:
            tail = "line %s" % line_pct(LINES[sense])
        else:
            tail = "no line measured"
        parts.append("%s %s (%s)" % (word(sense), span, tail))
    return ", ".join(parts)


def alert_text(judged: List[dict], runs: Optional[Dict[str, int]] = None) -> Optional[str]:
    """The watcher's alert sentence naming every body under its sense's line, or None when none is.
    runs: {sense: consecutive checks the sense has had a named body}."""
    hits = named(judged)
    if not hits:
        return None
    parts = []
    for j in hits:
        n = (runs or {}).get(j["sense"])
        parts.append("%s - %s against its line of %s%s" % (
            with_meaning(j["sense"]), bodies_text(j["bodies"], j["measured"]), line_pct(j["line"]),
            "" if not n else " (%d consecutive check%s)" % (n, "" if n == 1 else "s")))
    why = "; ".join("%s, %s" % (j["word"], j["reason"]) for j in hits)
    return ("DEAF SENSE ALERT (an alert, never a stop rule; training continues): " + "; ".join(parts)
            + ". Each named body gets almost the same token of that sense on every sampled model step (" + SCALE + "): "
            + CAUSES + ". Why each line: " + why + ".")


def card(readings, source_text: Optional[str] = None) -> dict:
    """The dashboard's card (title, value, notes), built from one record's readings."""
    title = "Deaf senses"
    if not isinstance(readings, dict) or not readings:
        return dict(title=title, value="Unmeasured",
                    notes=["This record carries no per-body collapse watch (token_temporal_std_bodies): the "
                           "trainer that wrote it predates the reading. A missing reading is not a deaf body."])
    judged = judge(readings)
    hits = named(judged)
    if hits:
        value = "; ".join("%s in %s" % (_cap(j["word"]) if i == 0 else j["word"], which_bodies(j["bodies"], j["measured"]))
                          for i, j in enumerate(hits))
    else:
        value = "No body under its sense's line"
    count = max([j["measured"] for j in judged] or [0])
    notes = ["A body is deaf in a sense when its token of that sense (what the sense's encoder hands the model "
             "each model step) is almost the same on every sampled model step over the collapse watch's 50 seconds: under that "
             "sense's line. The scale: " + SCALE + "."
             + (" Bodies are numbered 1 to %d." % count if count > 1 else "")]
    for j in hits:
        notes.append("%s: %s; the line is %s: %s." % (_cap(j["word"]), bodies_text(j["bodies"], j["measured"]),
                                                      line_pct(j["line"]), j["reason"]))
    summary = summary_text(readings)
    if summary:
        notes.append("Every sense on this record, lowest and highest body: " + summary + ".")
    notes.append("Two causes give a deaf reading: " + CAUSES + ".")
    if source_text:
        notes.append(source_text)
    return dict(title=title, value=value, notes=notes)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
