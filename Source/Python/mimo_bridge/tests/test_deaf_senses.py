# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The deaf-sense lines (mimo_bridge/deaf_senses.py), the hourly watcher's DEAF SENSE ALERT and the
dashboard's "Deaf senses" card (2026-09-25).

Every line and every number in its reason is recomputed here from the measurement it came from
(docs/gain_of_function_v3/measurements/proprio_token_bodies_2026-09-25.json: 10 H84 checkpoints x 8
bodies x 6 windows of 50 s), so a line moved without new evidence fails. The watcher and the card are
run on the real windows of checkpoint 411,125, where five bodies were deaf in body position, all eight
in balance and one in vision; the alert must name exactly those and must never stop the run.
"""
import collections
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

import pytest

from mimo_bridge import deaf_senses as DS
from mimo_bridge.tools.check_no_bare_decimals import (bare_decimals, doubled_sentences, long_numbers,
                                                       unanchored_numbers)

REPO = Path(__file__).resolve().parents[4]
BODIES = REPO / DS.MEASUREMENT
LINEAGE = REPO / "docs/gain_of_function_v3/measurements/proprio_token_lineage_2026-09-25.json"
WATCHER = REPO / "docs/h84_launch_2026-09-18/hourly_watch.py"
# the style guide's banned wording, as tests/test_dashboard_page_text.py holds every dashboard text to it
BANNED = re.compile(r"(?i)\b(honest\w*|truthful|barely|slightly|roughly|significantly|substantially|in band|"
                    r"healthy|armed|nominal|a bit)\b")

sys.path.insert(0, os.path.dirname(__file__))


def plain(text):
    """Every reader rule the watcher's and the dashboard's tests hold printed text to."""
    assert not bare_decimals(text), bare_decimals(text)
    assert not unanchored_numbers(text), unanchored_numbers(text)
    assert not long_numbers(text), long_numbers(text)
    assert not doubled_sentences(text), doubled_sentences(text)
    assert not BANNED.findall(text), BANNED.findall(text)


def rows():
    return json.loads(BODIES.read_text(encoding="utf-8"))["rows"]


def windows(sense):
    """Every 50-s window of every body at every checkpoint: (reading, step, body index)."""
    return [(v, r["step"], b) for r in rows() if sense in r["spread"]
            for b, ws in enumerate(r["spread"][sense]) for v in ws]


def six_window_means(sense):
    """Each body's mean over its six windows (300 s) at each checkpoint: (mean, step, body index)."""
    return [(sum(ws) / len(ws), r["step"], b) for r in rows() if sense in r["spread"]
            for b, ws in enumerate(r["spread"][sense])]


def one(value):
    """A reading as the reasons print it: a percentage with one decimal."""
    return "%.1f%%" % (100.0 * value)


def span(values):
    return "%s-%s" % (one(min(values)).rstrip("%"), one(max(values)))


# ------------------------------------------------------------------------- the lines and their evidence
def test_the_two_group_lines_sit_in_the_gap_of_the_measured_windows():
    """vision and body position: 480 windows each; the line lies between the deaf and the live
    six-window means with none between, and the reason's numbers are the measured ones."""
    for sense, low_edge, high_edge, n_low, n_mid, n_high in (("vision", 0.15, 0.25, 35, 12, 433),
                                                            ("proprioception", 0.15, 0.30, 108, 53, 319)):
        w = windows(sense)
        assert len(w) == 480
        assert sum(v < low_edge for v, _s, _b in w) == n_low
        assert sum(low_edge <= v < high_edge for v, _s, _b in w) == n_mid
        assert sum(v >= high_edge for v, _s, _b in w) == n_high
        line = DS.LINES[sense]
        means = six_window_means(sense)
        under = [v for v, _s, _b in means if v < line]
        over = [v for v, _s, _b in means if v >= line]
        assert under and over and max(under) < line <= min(over)
        # the six-window means leave a gap around the line: none within it
        assert not [v for v, _s, _b in means if max(under) < v < min(over)]
        assert span(under) in DS.REASONS[sense] and span(over) in DS.REASONS[sense], (span(under), span(over))
    # vision's deaf windows all come from the one body lying face down (index 2, the dashboard's body 3)
    assert {b for v, _s, b in windows("vision") if v < 0.15} == {2}
    assert {b for v, _s, b in six_window_means("vision") if v < DS.LINES["vision"]} == {2}
    assert len([1 for v, _s, _b in six_window_means("proprioception") if v < DS.LINES["proprioception"]]) == 18


def test_the_one_group_lines_are_the_lowest_window_rounded_down():
    for sense in ("tactile", "audio", "motor"):
        low = min(v for v, _s, _b in windows(sense))
        assert DS.LINES[sense] == math.floor(100 * low) / 100, (sense, low)
        assert "lowest %s" % one(low) in DS.REASONS[sense]
        # no body's six-window mean falls under it: nothing on H84 was deaf in these senses
        assert min(v for v, _s, _b in six_window_means(sense)) > DS.LINES[sense]


def test_the_balance_line_is_the_placeholders_lowest_reading_and_balance_reads_zero():
    placeholder = [v for v, _s, _b in windows("intermind")]
    assert len(placeholder) == 384 and one(min(placeholder)) == DS.line_pct(DS.LINES["vestibular"]) == "1.6%"
    assert "(%d fifty-second windows on H84)" % len(placeholder) in DS.REASONS["vestibular"]
    balance = [v for v, _s, _b in windows("vestibular")]
    assert len(balance) == 96 and max(balance) == 0.0


def test_the_body_position_verdict_holds_for_any_line_from_18_to_25_percent():
    """The docstring's claim about the 64-s readings of all 227 checkpoints."""
    lineage = [r for r in json.loads(LINEAGE.read_text(encoding="utf-8"))["rows"] if r.get("memory_events")]
    assert len(lineage) == 227
    p = lambda r: r["memory_events"]["senses"]["proprioception"]["time_std"]      # noqa: E731
    for line in (0.18, 0.20, 0.22, 0.25):
        before = [sum(v < line for v in p(r)) for r in lineage if r["step"] <= 344400]
        after = [sum(v < line for v in p(r)) for r in lineage if r["step"] >= 344525]
        assert (statistics.median(before), statistics.median(after)) == (2, 5), line


# ------------------------------------------------------------------------- judging
def test_bodies_are_numbered_from_one_and_the_placeholder_is_never_judged():
    readings = dict(proprioception=[0.5, 0.05, None, float("nan"), 0.3], intermind=[0.0, 0.0],
                    vestibular=[0.0, 0.0], voice=[0.001, 0.4])
    judged = {j["sense"]: j for j in DS.judge(readings)}
    assert set(judged) == {"proprioception", "vestibular"}              # no line for voice; intermind skipped
    assert judged["proprioception"]["bodies"] == [(2, 0.05)] and judged["proprioception"]["measured"] == 3
    assert DS.which_bodies(judged["vestibular"]["bodies"], 2) == "all 2 bodies"
    assert DS.bodies_text(judged["vestibular"]["bodies"], 2) == "all 2 bodies 0%"
    text = DS.summary_text(readings)
    assert "voice 0.1% to 40% (no line measured)" in text
    assert "the placeholder channel, not a sense, not judged" in text
    assert DS.which_bodies([(1, 0.1), (3, 0.1), (8, 0.1)], 8) == "bodies 1, 3 and 8"
    assert DS.alert_text(DS.judge(dict(motor=[0.6, 0.5]))) is None
    plain(text)


# ------------------------------------------------------------------------- the watcher and the card
def records_411125():
    """Six records, one per 50-s window of checkpoint 411,125: an hour's mean of them is each body's
    six-window mean."""
    row = next(r for r in rows() if r["step"] == 411125)
    return [dict(type="mimo", token_temporal_std_bodies={s: [row["spread"][s][b][k] for b in range(8)]
                                                         for s in row["mods"]}) for k in range(6)]


def watcher_module():
    spec = importlib.util.spec_from_file_location("h84_hourly_watch_deaf", WATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_watcher_names_every_deaf_body_and_sense_with_its_line_and_reason():
    m = watcher_module()
    w = records_411125()
    lines = m.deaf_sense_lines(w)
    assert len(lines) == 2, lines
    gauge, alert = lines
    assert gauge.startswith("  per-body collapse watch (hour means") and "(line 20%)" in gauge
    # the senses go by the registry's names (the style guide "ONE WORD PER THING"); the alert, which is read alone, gives
    # each sense it names its plain meaning, and that meaning is the opening of the explainer's own definition
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    assert DS.SENSE_WORDS == {s: s for s in DS.SENSE_MEANS}
    for sense, meaning in DS.SENSE_MEANS.items():
        assert SENSE_DEFINITIONS[sense].startswith(meaning), sense
    assert "vestibular 0% (line 1.6%)" in gauge and "tactile " in gauge and "motor " in gauge
    for old in ("touch", "hearing", "body position", "balance 0%", "moment"):
        assert old not in gauge and old not in alert.replace("the skin's sense of touch", ""), old
    prop, vest, vis = (DS.with_meaning(s) for s in ("proprioception", "vestibular", "vision"))
    assert prop == "proprioception (where each tracked body part is, how it is turned and how fast it moves)"
    assert alert.startswith("  DEAF SENSE ALERT (an alert, never a stop rule; training continues): " + prop + " - body 1 ")
    parts = alert.split("; ")
    body_position = next(p for p in parts if prop + " - " in p)
    assert re.findall(r"body (\d) ", body_position) == ["1", "2", "3", "6", "8"], body_position
    assert vest + " - all 8 bodies 0% against its line of 1.6% (1 consecutive check)" in alert
    vision = next(p for p in parts if p.startswith(vis + " - "))
    assert re.findall(r"body (\d) ", vision) == ["3"] and "against its line of 20%" in vision
    for quiet in ("tactile", "audio", "motor"):
        assert DS.with_meaning(quiet) + " - " not in alert
    for sense in ("proprioception", "vestibular", "vision"):
        assert DS.REASONS[sense] in alert
    assert "the sense's encoder stopped answering that body, or the body's input held still" in alert
    plain("\n".join(lines))
    # the next check counts on
    assert "(2 consecutive checks)" in m.deaf_sense_lines(w)[1]
    # a sense whose bodies recover resets its count
    live = [dict(r, token_temporal_std_bodies=dict(r["token_temporal_std_bodies"], vision=[0.4] * 8)) for r in w]
    assert vis + " - " not in m.deaf_sense_lines(live)[1]
    again = m.deaf_sense_lines(w)[1]
    assert vis + " - body 3 6.9% against its line of 20% (1 consecutive check)" in again
    assert vest + " - all 8 bodies 0% against its line of 1.6% (4 consecutive checks)" in again


def test_a_record_without_the_reading_prints_unmeasured_and_names_nothing():
    m = watcher_module()
    lines = m.deaf_sense_lines([dict(type="mimo"), dict(type="mimo", token_temporal_std=dict(vision=0.4))])
    assert lines == ["  per-body collapse watch: unmeasured - no record in the hour carries token_temporal_std_bodies "
                     "(the trainer's code predates it); a missing reading is not a deaf body"]
    # a body missing from some records is averaged over the records that carry it
    w = [dict(token_temporal_std_bodies=dict(vision=[0.4, 0.05])), dict(token_temporal_std_bodies=dict(vision=[0.4]))]
    assert "vision (what the two eyes see) - body 2 5.0% against its line of 20%" in m.deaf_sense_lines(w)[1]


def test_the_alert_never_stops_the_run(tmp_path, monkeypatch, capsys):
    """Run from run-hour 3 on a record whose bodies are deaf in body position, balance and vision (the
    known defects of 2026-09-25): the watcher prints the alert every check and writes no stop."""
    H = importlib.import_module("test_h84_hourly_watch")
    deaf = records_411125()[0]["token_temporal_std_bodies"]
    assert H.watch(tmp_path, monkeypatch, baseline=H.BASE, elapsed_s=4 * 3600, token_temporal_std_bodies=deaf) is None
    out = capsys.readouterr().out
    assert out.count("DEAF SENSE ALERT") == 3                           # every one of the 3 checks
    assert "(3 consecutive checks)" in out and "training continues" in out
    assert not (tmp_path / "stop_reason.json").exists() and not (tmp_path / "control.json").exists()
    printed = [l for l in out.splitlines() if not l.startswith(("Monitoring", "own baselines loaded"))]
    assert not [l for l in printed if "LINE FAILED" in l]
    # the watcher's own rule on everything it printed (three checks repeat their sentences by design)
    assert not bare_decimals("\n".join(printed)), bare_decimals("\n".join(printed))
    assert not unanchored_numbers("\n".join(printed)), unanchored_numbers("\n".join(printed))


def test_the_dashboard_card_names_the_same_bodies_in_plain_words():
    from mimo_bridge.tools import development_dashboard as dd
    w = records_411125()
    mean = {s: [sum(r["token_temporal_std_bodies"][s][b] for r in w) / len(w) for b in range(8)]
            for s in w[0]["token_temporal_std_bodies"]}
    record = dict(type="mimo", step=411125, timestamp=1_790_000_400, token_temporal_std_bodies=mean)
    cards = dd.observation_cards({}, record, {})
    titles = [c["title"] for c in cards]
    assert titles.index("Deaf senses") == titles.index("How the bodies lie") + 1
    card = cards[titles.index("Deaf senses")]
    assert card["value"] == "Proprioception in bodies 1, 2, 3, 6 and 8; vestibular in all 8 bodies; vision in body 3"
    text = " ".join([card["title"], card["value"], *card["notes"]])
    assert "Bodies are numbered 1 to 8." in text and "the line is 20%: " + DS.REASONS["vision"] in text
    assert "lineage step 411,125" in text
    plain(text)
    missing = dd.observation_cards({}, dict(type="mimo", step=5), {})
    card = next(c for c in missing if c["title"] == "Deaf senses")
    assert card["value"] == "Unmeasured" and "A missing reading is not a deaf body." in card["notes"][0]
    live = dict(record, token_temporal_std_bodies={s: [0.45] * 8 for s in ("vision", "tactile", "audio",
                                                                            "proprioception", "motor")})
    card = next(c for c in dd.observation_cards({}, live, {}) if c["title"] == "Deaf senses")
    assert card["value"] == "No body under its sense's line"
    plain(" ".join([card["title"], card["value"], *card["notes"]]))
