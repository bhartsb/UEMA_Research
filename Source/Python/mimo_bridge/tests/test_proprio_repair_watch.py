# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The proprioception token repair's kill and acceptance lines (docs/gain_of_function_v4/proprio_repair_watch.py,
version 6: every body line reads the token's response per unit of the body's own posture movement; the third review's
findings fixed, proprio_token_review3_2026-09-26.md) and their enforcement by the H84 hourly watcher (review finding 5
of docs/gain_of_function_v4/agents/proprio_token_review_2026-09-25.md; the second review's findings,
proprio_token_review2_2026-09-26.md; record sections 43 and 45).

What is pinned: a run the switch does not reach is not judged; each kill line holds on a record that stays within it
and crosses on one that leaves it (K1 on the first report, on a foreign switch, and on a lesson that runs with another
line, another DEAF or other skipped bodies than the lines file measured; K2 on a body's spread three reports running
and on its jitter share minus the posture's own after the lesson's start, against the difference line shifted into the
lesson's units by every judged body at the start, never on the gain the lesson adds nor on a change of behaviour that
moves both shares, and judged even when every body starts under LINE; K3 and K5 at report 21 and again over the 20
reports after the lesson's start; K4; K6 at every window after the start in log units, stopping only when the lesson's
own attributed change accounts for the fall); body 3 and a skipped body never trip a line on bodies; the balance sense
is watched only when its token is alive with enough history; A1 judges the deaf bodies of the start at three windows
after the horizon on the line where a lifted body settles (the deaf/live line under its noise), the attributed lift,
the posture-only ratio, the gain-matched content and the fatigue answer (at the hold's line: the live bodies' largest
at the start, the one the lesson recorded), with the success metric and its held-out reading beside it; A2 at the
operator's deaf/live line on the horizon checkpoint, converted by the measured log gap, with the generalization reading
beside it; the fatigue hold is printed with its line and the bodies it pulls back; the distance from the pre-deploy
state is a share, never a pass line; a missing lines file makes its lines wait; the lines and A2 files are built from
version 6's pause results; a crossing whose stop was taken is not stopped again; the
hourly watcher's 5-minute check writes the stop through its own main loop; every printed line obeys the reader
rules. Version 7 (the record's section 47.5; the review proprio_token_v7_review_2026-09-27.md and the coordinator's
decisions): K1 needs version 7's switch record; A1 (ii) judges the attribution divided by the over-read rule LS version
7's pairs measured for the target's class, and is a reading until they have run; A1 (vi) reads the hold's line in
force; K6 states its limit on bodies the lesson never pushed; the hold's readings (its line now, the live bodies'
answer now, its share of recent steps) are printed; the lines file carries the over-read per class from every measure
result; A2 prints the planner's goal-pick reading."""
import importlib
import importlib.util
import json
import math
import runpy
import sys
import time
from pathlib import Path

import pytest

from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, unanchored_numbers

REPO = Path(__file__).resolve().parents[4]
WATCH = REPO / "docs" / "gain_of_function_v4" / "proprio_repair_watch.py"
WATCHER = REPO / "docs" / "h84_launch_2026-09-18" / "hourly_watch.py"
N = 50000
S = N + 100                     # the lesson's start in these records (its first full window)
MOVE = 0.07                     # every body's posture movement in these records
START = [1.0, 3.5, 0.3, 3.5, 3.5, 3.5, 3.5, 1.2]      # bodies 1 and 8 deaf (under DEAF 2.0), body 3 face down


@pytest.fixture()
def W():
    spec = importlib.util.spec_from_file_location("proprio_repair_watch_t", WATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jitter(ratio, share=0.3):
    """The jitter that gives `share` of the token's variance at the response ratio x MOVE."""
    return [2 * share * (r * MOVE) ** 2 for r in ratio]


def start_block(ratio=None, share=0.3, content=None, pushed=None, posture_share=0.1, fatigue=None, step=S,
                fatigue_line="live"):
    """The lesson's start as version 6 records it. fatigue_line "live": the hold's line the lesson records (the largest
    answer of the live bodies: judged, not engaged); None: a start that records none."""
    ratio = list(ratio or START)
    shares = share if isinstance(share, list) else [share] * 8
    fat = list(fatigue or [0.16] * 8)
    engaged = [r < 2.0 and b != 2 for b, r in enumerate(ratio)]
    live = [b != 2 and not e for b, e in enumerate(engaged)]
    block = dict(step=step, ratio=ratio, ratio_posture_only=[0.96 * r for r in ratio], spread=[r * MOVE for r in ratio],
                 posture_spread=[MOVE] * 8, jitter=[2 * x * (r * MOVE) ** 2 for x, r in zip(shares, ratio)],
                 jitter_share=list(shares), posture_jitter_share=[posture_share] * 8,
                 content=list(content or [0.3] * 8), content_gain_matched=list(content or [0.3] * 8),
                 fatigue_answer=fat,
                 pushed=list(pushed if pushed is not None else [r < 3.0 and b != 2 for b, r in enumerate(ratio)]),
                 engaged=engaged, live=live, line=3.0, deaf=2.0, skip=[3])
    if fatigue_line == "live":
        block["fatigue_line"] = max(f for f, ok in zip(fat, live) if ok)
    elif fatigue_line is not None:
        block["fatigue_line"] = fatigue_line
    return block


def rec(step, ratio=None, sps=1.0, spread=None, repair=True, nan=0, tts_vest=0.0, vest_ratio=0.004, base=None,
        start=True, attributed=None, posture_only=None, content=None, share=0.3, senses=None, posture_share=0.1,
        gm=None, fatigue=None, held_out=None, pushed_ever=None, held=None, fatigue_line=None, fatigue_p=None,
        fatigue_line_step=None, hold_any=None):
    senses = dict(senses or {"vision": 1.0, "tactile": 1.0, "audio": 1.0})
    senses["vestibular"] = vest_ratio
    r = dict(base or {})
    r.update(type="mimo", step=step, timestamp=1e9 + step, nan_steps=nan, model_steps_per_s=sps,
             next_frame_norm=dict(steps=100, ratio=senses),
             token_temporal_std=dict((base or {}).get("token_temporal_std") or {}, vestibular=tts_vest),
             token_temporal_std_bodies=dict(proprioception=spread or [0.4] * 8))
    if repair == "filling":
        r["proprio_token_repair"] = dict(samples=100)             # the window not yet full: no reading, no start
    elif repair:
        ratio = list(ratio or START)
        shares = share if isinstance(share, list) else [share] * 8
        pshares = posture_share if isinstance(posture_share, list) else [posture_share] * 8
        block = dict(ratio=ratio, ratio_posture_only=posture_only or [0.96 * x for x in ratio],
                     spread=[x * MOVE for x in ratio], posture_spread=[MOVE] * 8,
                     jitter=[2 * s * (x * MOVE) ** 2 for s, x in zip(shares, ratio)],
                     jitter_share=list(shares), posture_jitter_share=list(pshares),
                     content=content or [0.3] * 8, content_gain_matched=gm or content or [0.3] * 8,
                     fatigue_answer=fatigue or [0.16] * 8,
                     attributed=attributed or [0.0] * 8, below_line=2, lesson=0.5,
                     pushed_ever=pushed_ever or [b in (0, 7) for b in range(8)], held=held or [False] * 8,
                     applied=dict(share_of_clipped_gradient_median=0.02))
        if held_out is not None:
            block["held_out"] = held_out
        # version 7's hold readings: the line in force, its step, every moving body's posture-only answer, the share of
        # the steps since the report before in which the hold acted
        for key, v in (("fatigue_line", fatigue_line), ("fatigue_answer_posture_only", fatigue_p),
                       ("fatigue_line_step", fatigue_line_step), ("hold_any_share_since_gauge", hold_any)):
            if v is not None:
                block[key] = v
        if start:
            block["start"] = start if isinstance(start, dict) else start_block()
        r["proprio_token_repair"] = block
    return r


LINES_DOC = dict(version="proprio_token_repair_v7", N=N, LINE=3.0, line_rebuilt=3.0, DEAF=2.0, deaf_rebuilt=2.0,
                 SKIP=[3], K2_share_diff=0.35, switch_share_diff=[0.2] * 8, switch_ratio=list(START),
                 switch_content=[0.3] * 8, switch_fatigue=[0.16] * 8, fatigue_sigma_hi=0.03,
                 window_sigma_log=0.1, window_sigma_log_hi=0.12, k6_fall_log=0.33, content_sigma_hi=0.05,
                 A1_horizon=1000.0,
                 # version 5's K2 line, kept so the same records judge the version-5 code in the fail-first runs
                 K2_share=0.6, switch_share=[0.3] * 8)


def make_runs(tmp_path, after, before=None, switch=True, lines=True, version="proprio_token_repair_v7", line=3.0,
              skip=(3,), lines_extra=None, deaf=2.0):
    exp = tmp_path / "experiments"
    parent, deploy = exp / "H84_parent", exp / "H84_deploy"
    (parent / "checkpoints" / ("ckpt_step_%d" % N)).mkdir(parents=True)
    deploy.mkdir(parents=True)
    before = before if before is not None else [rec(N - 100 * k, repair=False) for k in range(25, -1, -1)]
    (parent / "replay_training_stats.jsonl").write_text("\n".join(json.dumps(r) for r in before) + "\n",
                                                        encoding="utf-8")
    (deploy / "args.json").write_text(json.dumps(dict(resume=str(parent / "checkpoints" / ("ckpt_step_%d" % N)))),
                                      encoding="utf-8")
    body = []
    if switch:
        body.append(dict(type="proprio_token_repair", step=N, timestamp=1e9 + N,
                         switch=dict(version=version, line=line, deaf=deaf, skip=list(skip), strength=0.02)))
    body += after
    (deploy / "replay_training_stats.jsonl").write_text("\n".join(json.dumps(r) for r in body) + "\n",
                                                        encoding="utf-8")
    if lines:
        doc = dict(LINES_DOC)
        doc.update(lines_extra or {})
        (deploy / "proprio_repair_lines.json").write_text(json.dumps(doc), encoding="utf-8")
    return deploy


def after_reports(n=25, **kw):
    return [rec(N + 100 * (k + 1), **kw) for k in range(n)]


def judged(W, deploy):
    return W.evaluate(deploy, W.StatsReader(), lines_file=deploy / "none.json")


def printed(W, deploy):
    return "\n".join(W.lines(deploy, W.StatsReader(), lines_file=deploy / "none.json"))


def readable(text):
    assert not bare_decimals(text), bare_decimals(text)
    assert not unanchored_numbers(text), unanchored_numbers(text)


def test_a_run_the_switch_does_not_reach_is_not_judged(W, tmp_path):
    deploy = make_runs(tmp_path, after_reports(), switch=False)
    assert W.evaluate(deploy, W.StatsReader()) is None
    assert W.lines(deploy, W.StatsReader()) == []


def test_every_line_holds_on_a_record_within_it(W, tmp_path):
    deploy = make_runs(tmp_path, after_reports(70))                  # the start and one window after it
    ev = judged(W, deploy)
    assert ev["why"] is None, ev["judgement"]["lines"]
    L = ev["judgement"]["lines"]
    assert {k: L[k]["state"] for k in ("K1", "K2", "K3", "K4", "K5", "K6")} == dict.fromkeys(
        ("K1", "K2", "K3", "K4", "K5", "K6"), "held")
    assert ev["S"] == S and L["K6"]["windows"] == 1
    text = printed(W, deploy)
    assert "body 3, face down since Sept 19, is left out" in text and "its start read at step 50,100" in text
    assert "the vestibular token is dead" in text and "response per unit of its own posture movement" in text
    readable(text)


def test_k1_crosses_on_the_first_report_a_foreign_switch_and_unmeasured_settings(W, tmp_path):
    first = [rec(N + 100, senses={"vision": 4.3, "tactile": 1.0, "audio": 1.0})] + after_reports()[1:]
    ev = judged(W, make_runs(tmp_path / "a", first))
    assert ev["line"] == "K1" and "vision" in ev["why"]
    readable(ev["why"])
    ev = judged(W, make_runs(tmp_path / "b", after_reports(), version="proprio_token_repair_v5"))
    assert ev["line"] == "K1" and "not this version" in ev["why"], "version 5's lesson has no hinge"
    ev = judged(W, make_runs(tmp_path / "b6", after_reports(), version="proprio_token_repair_v6"))
    assert ev["line"] == "K1" and "not this version" in ev["why"], "version 6's hold reads its line once"
    ev = judged(W, make_runs(tmp_path / "c", after_reports(repair=False)))
    assert ev["line"] == "K1" and "no proprio_token_repair block" in ev["why"]
    ev = judged(W, make_runs(tmp_path / "d", after_reports(), line=3.5))
    assert ev["line"] == "K1" and "--proprio-token-line 3.5 where" in ev["why"]
    readable(ev["why"])
    ev = judged(W, make_runs(tmp_path / "e", after_reports(), skip=()))
    assert ev["line"] == "K1" and "it skips bodies none where the lines file skips bodies 3" in ev["why"]
    ev = judged(W, make_runs(tmp_path / "g", after_reports(), deaf=2.5))
    assert ev["line"] == "K1" and "--proprio-token-deaf 2.5 where" in ev["why"], "version 6: the hinge's DEAF"
    readable(ev["why"])
    assert judged(W, make_runs(tmp_path / "f", after_reports(), line=3.5, lines=False))["why"] is None, \
        "without a lines file the settings wait to be checked; the rest of K1 is judged"


def test_k2_needs_three_reports_running_and_never_counts_body_3_or_a_skipped_body(W, tmp_path):
    hot = [0.4] * 8
    hot[1] = 0.8
    rows = after_reports(2) + [rec(N + 300, spread=hot), rec(N + 400, spread=hot), rec(N + 500)] + \
        [rec(N + 100 * k) for k in range(6, 26)]
    assert judged(W, make_runs(tmp_path / "a", rows))["why"] is None, "two reports running do not stop"
    rows3 = after_reports(2) + [rec(N + 100 * k, spread=hot) for k in (3, 4, 5)] + \
        [rec(N + 100 * k) for k in range(6, 26)]
    ev = judged(W, make_runs(tmp_path / "b", rows3))
    assert ev["line"] == "K2" and "body 2" in ev["why"] and "above 75%" in ev["why"]
    readable(ev["why"])
    only3 = [0.4] * 8
    only3[2] = 0.9
    rows_b3 = [rec(N + 100 * k, spread=only3, share=[0.3, 0.3, 0.99, 0.3, 0.3, 0.3, 0.3, 0.3]) for k in range(1, 26)]
    assert judged(W, make_runs(tmp_path / "c", rows_b3))["why"] is None, "body 3 is left out"
    skip4 = [0.4] * 8
    skip4[3] = 0.9
    rows_s = [rec(N + 100 * k, spread=skip4) for k in range(1, 26)]
    assert judged(W, make_runs(tmp_path / "s", rows_s, skip=(3, 4), lines_extra=dict(SKIP=[3, 4])))["why"] is None, \
        "a body the lesson skips is judged by no line"
    share = [0.3] * 8
    share[7] = 0.7
    ev = judged(W, make_runs(tmp_path / "d", [rec(N + 100 * k, share=share) for k in range(1, 26)]))
    assert ev["line"] == "K2" and "body 8" in ev["why"] and "the posture's own share" in ev["why"]


def test_k2s_jitter_part_judges_the_tokens_share_over_the_postures_own_shifted_by_every_judged_body(W, tmp_path):
    """Version 6 (the third review's MAJOR): K2's jitter part judges the token's jitter share minus the posture's own on
    the same window, against K2_SHARE_DIFF shifted into the lesson's units by the median over EVERY judged body of the
    start's difference minus the switch's calibrated difference. Pinned: a gain leaves it (a tripled ratio at the same
    shares holds); the shift moves the line; a change of behaviour that raises both shares by 0.35 holds, where version
    5's line on the token's share alone (0.6 here) stopped the run; with every judged body under LINE at the start (as
    at 380,550) a noise-like token still stops the run, where version 5 found no body to rescale by and judged nothing;
    a body whose start difference is over the line is judged by the spread part alone."""
    tripled = [3.0 * r for r in START]
    ev = judged(W, make_runs(tmp_path / "gain", [rec(N + 100 * k, ratio=tripled) for k in range(1, 26)]))
    assert ev["why"] is None and ev["judgement"]["lines"]["K2"]["diff_line"] == pytest.approx(0.35)
    assert ev["judgement"]["lines"]["K2"]["diff_shift"] == pytest.approx(0.0)
    share = [0.3] * 8
    share[3] = 0.62                                   # body 4: 0.62 - 0.1 = 0.52, over 0.35, under 0.35 + 0.2
    extra = dict(switch_share_diff=[0.0] * 8)         # the calibration read 0.2 less: the line moves to 0.55
    ev = judged(W, make_runs(tmp_path / "a", [rec(N + 100 * k, share=share) for k in range(1, 26)], lines_extra=extra))
    assert ev["why"] is None and ev["judgement"]["lines"]["K2"]["diff_line"] == pytest.approx(0.55)
    ev = judged(W, make_runs(tmp_path / "b", [rec(N + 100 * k, share=share) for k in range(1, 26)]))
    assert ev["line"] == "K2" and "body 4" in ev["why"], "the control: unshifted, 0.52 crosses 0.35"
    # a change of behaviour: the posture itself gets jumpier and the token with it (the Sept 23 deploy)
    drift = [rec(N + 100 * k, share=0.65, posture_share=0.45) for k in range(1, 26)]
    ev = judged(W, make_runs(tmp_path / "drift", drift))
    assert ev["why"] is None, "both shares rose by 0.35: the difference stayed 0.2"
    # every judged body under LINE at the start, then one body's token turns noise-like
    under = [1.0, 2.7, 0.3, 2.5, 2.6, 2.7, 2.8, 1.2]
    noisy = [0.3] * 8
    noisy[3] = 0.95
    rows = [rec(N + 100 * k, ratio=under, share=noisy, start=start_block(ratio=under)) for k in range(1, 26)]
    ev = judged(W, make_runs(tmp_path / "under", rows))
    assert ev["line"] == "K2" and "body 4" in ev["why"], "judged although no body started over LINE"
    readable(ev["why"])
    filling = [rec(N + 100 * k, share=share, start=False) for k in range(1, 26)]
    ev = judged(W, make_runs(tmp_path / "c", filling))
    assert ev["why"] is None and "not read yet" in ev["judgement"]["lines"]["K2"]["diff_why"]
    hot_start = start_block(share=[0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3])
    rows = [rec(N + 100 * k, share=[0.3, 0.3, 0.3, 0.8, 0.3, 0.3, 0.3, 0.3], start=hot_start) for k in range(1, 26)]
    ev = judged(W, make_runs(tmp_path / "d", rows))
    assert ev["why"] is None and ev["judgement"]["lines"]["K2"]["diff_exempt"] == [4]


def test_k3_k4_and_k5(W, tmp_path):
    rows = [rec(N + 100)] + [rec(N + 100 * k, senses={"vision": 1.0, "tactile": 1.6, "audio": 1.0})
                             for k in range(2, 26)]
    ev = judged(W, make_runs(tmp_path / "a", rows))
    assert ev["line"] == "K3" and "tactile (the skin's sense of touch)" in ev["why"]
    rows = after_reports(3) + [rec(N + 400, nan=1)] + [rec(N + 100 * k) for k in range(5, 26)]
    assert judged(W, make_runs(tmp_path / "b", rows))["line"] == "K4"
    rows = [rec(N + 100)] + [rec(N + 100 * k, sps=0.7) for k in range(2, 26)]
    assert judged(W, make_runs(tmp_path / "c", rows))["line"] == "K5"


def test_k3_and_k5_judge_the_other_senses_and_the_step_rate_again_after_the_lessons_start(W, tmp_path):
    """Version 6 (the third review's MAJOR): the lesson acts only from its start S (the window fills 6,000 steps after
    the switch), after report 21, where K3 and K5 closed. They are judged again over the 20 full reports after S
    against the 20 full reports before S (the flag on, the lesson not yet acting), at the same W20 and 0.8 lines;
    every later 20-report block against the same baseline is a finding, never a stop. The control: the same records
    with touch's error unchanged hold."""
    S2 = N + 2100

    def rows(touch_after=1.0, sps_after=1.0, later=None):
        out = []
        for k in range(1, 70):
            step = N + 100 * k
            if step <= S2:
                out.append(rec(step, repair="filling") if step < S2 else rec(step, start=start_block(step=S2)))
            else:
                t = later(k) if later else touch_after
                out.append(rec(step, start=start_block(step=S2), sps=sps_after if later is None else 1.0,
                               senses={"vision": 1.0, "tactile": t, "audio": 1.0}))
        return out
    ev = judged(W, make_runs(tmp_path / "held", rows()))
    k3 = ev["judgement"]["lines"]["K3"]
    assert ev["why"] is None and k3["lesson"]["state"] == "held" and k3["relaunch"]["state"] == "held"
    ev = judged(W, make_runs(tmp_path / "touch", rows(touch_after=1.6)))
    assert ev["line"] == "K3" and "tactile (the skin's sense of touch)" in ev["why"] and "after the lesson's start" in ev["why"]
    readable(ev["why"])
    ev = judged(W, make_runs(tmp_path / "slow", rows(sps_after=0.7)))
    assert ev["line"] == "K5" and "before the lesson's start" in ev["why"]
    ev = judged(W, make_runs(tmp_path / "later", rows(later=lambda k: 1.6 if k >= 43 else 1.0)))
    assert ev["why"] is None and any("tactile" in f and "not a stop" in f for f in ev["judgement"]["findings"])


def _windows(n_windows, ratio_at=None, attributed_at=None):
    out = []
    for step in range(N + 100, S + 6000 * n_windows + 1, 100):
        k = (step - S) // 6000
        out.append(rec(step, ratio=(ratio_at(k) if ratio_at else None),
                       attributed=(attributed_at(k) if attributed_at else None)))
    return out


def test_k6_judges_every_window_in_log_units_and_stops_only_when_the_lesson_accounts_for_the_fall(W, tmp_path):
    """K6 (version 5): a judged body's ln ratio at a window W_k more than K6_FALL_LOG (0.33 here: to 72 in 100 of the
    start) under the start's, AND the lesson's attributed change of ln ratio negative and at least half the fall. The
    control: the same fall with no attributed share is printed as production's and never stops the run."""
    def lost_at_3(k):
        r = list(START)
        if k >= 3:
            r[3] = 2.3                                       # body 4: ln(3.5 / 2.3) = 0.42, past 0.33, in window 3
        return r

    def lesson_took(k):
        a = [0.0] * 8
        if k >= 3:
            a[3] = -0.3                                     # the lesson's own steps: more than half of 0.42
        return a
    ev = judged(W, make_runs(tmp_path / "a", _windows(4, lost_at_3, lesson_took)))
    assert ev["line"] == "K6" and "body 4" in ev["why"] and "window 3" in ev["why"], "caught at window 3, not 1"
    readable(ev["why"])
    ev = judged(W, make_runs(tmp_path / "b", _windows(4, lost_at_3)))
    assert ev["why"] is None and ev["judgement"]["lines"]["K6"]["windows"] == 4
    finding = " ".join(ev["judgement"]["findings"])
    assert "body 4" in finding and "production's training, not the lesson" in finding
    # K6's limit (the version-7 review's finding 5): body 4 was never pushed, where the attribution ran 11-18 times
    # the change rule LS measured: the finding and the hourly line say K6 cannot tell the lesson's loss there
    assert "K6's limit: the lesson never pushed this body" in finding
    text = printed(W, make_runs(tmp_path / "c", _windows(4, lost_at_3)))
    assert "finding (not a stop)" in text
    assert "K6's limit" in text and "11 to 18 times the change on bodies it never pushed" in text
    readable(text)
    ev = judged(W, make_runs(tmp_path / "d", _windows(4, lambda k: [1.0, 3.5, 0.3, 3.0, 3.5, 3.5, 3.5, 1.2],
                                                         lesson_took)))
    assert ev["why"] is None and not ev["judgement"]["findings"], "ln(3.5 / 3.0) = 0.15: within the noise"
    ev = judged(W, make_runs(tmp_path / "e", [rec(N + 100 * k, repair="filling") for k in range(1, 30)]))
    assert ev["judgement"]["lines"]["K6"]["state"] == "waiting" and "start" in ev["judgement"]["lines"]["K6"]["why"]
    ev = judged(W, make_runs(tmp_path / "f", after_reports(30)))
    assert ev["judgement"]["lines"]["K6"]["state"] == "waiting" and "window" in ev["judgement"]["lines"]["K6"]["why"]

    def body3(k):
        r = list(START)
        r[2] = 0.01
        return r

    def took3(k):
        a = [0.0] * 8
        a[2] = -3.0
        return a
    assert judged(W, make_runs(tmp_path / "g", _windows(2, body3, took3)))["why"] is None, "body 3 never counts"


def test_the_balance_sense_is_watched_only_alive_with_its_own_measured_wander(W, tmp_path):
    ev = judged(W, make_runs(tmp_path / "dead", after_reports()))
    assert "dead" in ev["judgement"]["balance"]["why"]
    alive_before = [rec(N - 100 * k, repair=False, tts_vest=0.3, vest_ratio=0.9) for k in range(300, -1, -1)]
    alive_after = [rec(N + 100 * k, tts_vest=0.3, vest_ratio=0.9) for k in range(1, 26)]
    ev = judged(W, make_runs(tmp_path / "alive", alive_after, before=alive_before))
    bal = ev["judgement"]["balance"]
    assert bal["why"] is None and bal["W1"] == pytest.approx(1.0) and bal["W20"] == pytest.approx(1.0)
    assert ev["why"] is None
    jump = [rec(N + 100, tts_vest=0.3, vest_ratio=1.5)] + alive_after[1:]
    ev = judged(W, make_runs(tmp_path / "jump", jump, before=alive_before))
    assert ev["line"] == "K1" and "balance" in ev["why"]
    short = [rec(N - 100 * k, repair=False, tts_vest=0.3, vest_ratio=0.9) for k in range(50, -1, -1)]
    ev = judged(W, make_runs(tmp_path / "short", alive_after, before=short))
    assert "200 needed" in ev["judgement"]["balance"]["why"]


LIFTED = [2.9, 3.5, 0.3, 3.5, 3.5, 3.5, 3.5, 2.95]           # bodies 1 and 8 at the live line (3.0) within the margin
PUSHED = [0.8, 0, 0, 0, 0, 0, 0, 0.7]                        # the lesson's attributed lifts (ln ratio)


OVER_ONE = dict(attribution_over_read=dict(classes={"deaf region": 1.0, "live region": 1.0}))


def _a1_run(W, tmp_path, name, end_ratio, attributed, posture_only=None, content=None, start=None, gm=None,
            fatigue=None, extra=OVER_ONE, fatigue_line=None, fatigue_p=None, printed_too=False):
    """The start (bodies 1 and 8 under DEAF), then reports through the horizon (the floor of 10,000) and three windows
    (18,000 steps) reading end_ratio. extra: the lines file's additions (by default rule LS version 7's over-read of the
    attribution measured at 1 for both classes, so (ii) is judged on the attribution itself)."""
    st = start or start_block()
    last = S + 10000 + 18000
    rows = [rec(step, ratio=end_ratio, attributed=attributed, posture_only=posture_only, content=content, start=st,
                gm=gm, fatigue=fatigue, fatigue_line=fatigue_line, fatigue_p=fatigue_p)
            for step in range(N + 100, last + 1, 100)]
    deploy = make_runs(tmp_path / name, rows, lines_extra=extra)
    got = judged(W, deploy)["judgement"]["acceptance"]
    return (got, printed(W, deploy)) if printed_too else got


def test_a1_needs_the_settled_line_the_lessons_lift_the_posture_only_ratio_the_content_and_the_fatigue_answer(
        W, tmp_path):
    """A1 (version 6): the targets are the judged bodies under DEAF at the start (bodies 1 and 8 here); at three windows
    after the start + the horizon each passes on (i) the mean ln ratio at or above ln DEAF - 2 SIGMA_LOG_HI / sqrt 3
    (where a body released at LINE settles: engaged again under DEAF), (ii) the lesson's attributed lift at least half
    its log shortfall, (iii) the posture-only ratio at the same line, (v) the GAIN-MATCHED content no lower than the
    start's content minus 2 CONTENT_SIGMA_HI / sqrt 3, (vi) the fatigue answer no higher than the live bodies' at the
    start plus 2 FATIGUE_SIGMA_HI / sqrt 3; (iv) the pooled log rise at least half the pooled log shortfall. The
    controls: no attributed lift, a lift the posture-only ratio does not share, a lift whose gain-matched content falls
    while its plain content rises (movement without posture information: version 5 passed it), a lift that answers
    fatigue unlike the live bodies (version 5 passed it), a body short of the line; and a body settled between the
    line and LINE passes (version 5 judged it at the live level and failed it)."""
    A = _a1_run(W, tmp_path, "met", LIFTED, PUSHED, content=[0.35] * 8)
    assert A["A1_targets"] == [1, 8] and A["A1"]["state"] == "met", A["A1"]
    assert A["A1"]["margin_log"] == pytest.approx(2 * 0.12 / 3 ** 0.5)
    assert A["A1"]["line"] == pytest.approx(2.0 * math.exp(-2 * 0.12 / 3 ** 0.5))
    settled = [2.2, 3.5, 0.3, 3.5, 3.5, 3.5, 3.5, 2.25]              # over 1.74 (DEAF's line), under 2.61 (LINE's)
    As = _a1_run(W, tmp_path, "settled", settled, PUSHED)
    assert As["A1"]["state"] == "met", ("a body settled between DEAF and LINE is lifted", As["A1"])
    plant = _a1_run(W, tmp_path, "plant", LIFTED, PUSHED, content=[0.35] * 8, gm=[0.15] * 8)
    assert plant["A1"]["state"] == "missed" and plant["A1"]["bodies"][1]["follows_posture"] is False, \
        "the plain content rose with the amplitude; the gain-matched content fell: movement without posture information"
    tired = _a1_run(W, tmp_path, "tired", LIFTED, PUSHED, fatigue=[0.35, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.35])
    assert tired["A1"]["state"] == "missed" and tired["A1"]["bodies"][1]["answers_fatigue_as_live"] is False
    assert tired["A1"]["bodies"][1]["fatigue_line"] == pytest.approx(0.16 + 2 * 0.03 / 3 ** 0.5)
    b1 = A["A1"]["bodies"][1]
    assert b1["reached"] and b1["lifted_by_lesson"] and b1["answers_posture"] and b1["follows_posture"]
    assert b1["gap_closed_to_line"] == pytest.approx((2.9 - 1.0) / (3.0 - 1.0))
    assert A["A1"]["live_level"] == pytest.approx(3.5)
    assert A["A1"]["pooled_gap_closed_to_live_now"] == pytest.approx(((2.9 + 2.95) / 2 - 1.1) / (3.5 - 1.1))
    noise = _a1_run(W, tmp_path, "noise", LIFTED, [0.0] * 8)
    assert noise["A1"]["state"] == "missed" and noise["A1"]["missed"] == [1, 8], "no attributed lift: not a pass"
    flat = _a1_run(W, tmp_path, "flat", LIFTED, PUSHED, posture_only=[0.96 * r for r in START])
    assert flat["A1"]["state"] == "missed" and flat["A1"]["bodies"][1]["answers_posture"] is False
    noisy = _a1_run(W, tmp_path, "noisy", LIFTED, PUSHED, content=[0.2] * 8, gm=[0.2] * 8)
    assert noisy["A1"]["state"] == "missed" and noisy["A1"]["bodies"][1]["follows_posture"] is False
    low = _a1_run(W, tmp_path, "low", [1.5, 3.5, 0.3, 3.5, 3.5, 3.5, 3.5, 2.95], PUSHED)
    assert low["A1"]["missed"] == [1]
    rows_w = [rec(N + 100 * k, ratio=LIFTED) for k in range(1, 200)]
    assert judged(W, make_runs(tmp_path / "wait", rows_w))["judgement"]["acceptance"]["A1"]["state"] == "waiting"


def test_a1_ii_judges_the_attribution_divided_by_the_over_read_the_pairs_measured_for_the_targets_class(W, tmp_path):
    """The review's finding 5 and the coordinator's decision F: the attribution over-read the measured lift 1.4 to 3.4
    times in rule LS's re-fit, so A1 (ii) judges the attribution divided by the over-read rule LS version 7's pairs
    measured for the target's class (the lines file's attribution_over_read; the live region is a target at or over DEAF
    x exp(-2 SIGMA_LOG), within one window's noise twice of DEAF, the deaf region one under it); a class the pairs could
    not bound cannot pass; until the pairs have run (no over-read in the lines file) (ii) prints as a reading and decides
    nothing. Here bodies 1 and 8 start at 1.0 and 1.2 (the deaf region: under 2.0 x exp(-0.2) = 1.64) and the lesson's
    attribution, 0.8 and 0.7, is over half their shortfall (0.55 and 0.46) but not over twice it. Fails on 67335783f
    (it judged the attribution itself, and with no measured over-read it still required it)."""
    over2 = dict(attribution_over_read=dict(classes={"deaf region": 2.0, "live region": 1.0}))
    A = _a1_run(W, tmp_path, "over2", LIFTED, PUSHED, content=[0.35] * 8, extra=over2)
    b1 = A["A1"]["bodies"][1]
    assert A["A1"]["state"] == "missed" and A["A1"]["missed"] == [1, 8], A["A1"]
    assert b1["attributed_class"] == "deaf region" and b1["over_read"] == 2.0
    assert b1["attributed_corrected"] == pytest.approx(0.4) and b1["lifted_by_lesson"] is False
    live = dict(attribution_over_read=dict(classes={"deaf region": 1.0, "live region": 2.0}))
    A = _a1_run(W, tmp_path, "live2", LIFTED, PUSHED, content=[0.35] * 8, extra=live)
    assert A["A1"]["state"] == "met", "the deaf region's over-read (1.0) applies to these targets"
    near = start_block(ratio=[1.8, 3.5, 0.3, 3.5, 3.5, 3.5, 3.5, 1.2])      # body 1 in the live region (1.64-2.0)
    A = _a1_run(W, tmp_path, "near", LIFTED, [0.4, 0, 0, 0, 0, 0, 0, 0.7], content=[0.35] * 8, start=near, extra=live)
    assert A["A1"]["bodies"][1]["attributed_class"] == "live region" and A["A1"]["missed"] == [1], \
        "0.4 / 2.0 = 0.2 under half of ln(3.0 / 1.8) = 0.26"
    unb = dict(attribution_over_read=dict(classes={"deaf region": "unbounded"}))
    A = _a1_run(W, tmp_path, "unb", LIFTED, PUSHED, content=[0.35] * 8, extra=unb)
    assert A["A1"]["state"] == "missed" and A["A1"]["bodies"][1]["lifted_by_lesson_judged"] is True
    A, text = _a1_run(W, tmp_path, "reading", LIFTED, [0.0] * 8, content=[0.35] * 8, extra=None, printed_too=True)
    b1 = A["A1"]["bodies"][1]
    assert A["A1"]["state"] == "met" and b1["lifted_by_lesson"] is None and b1["lifted_by_lesson_judged"] is False
    assert "a reading only" in b1["attributed_reading"]
    assert "a reading only, deciding nothing, until rule LS version 7's pairs have run" in text
    readable(text)
    _A, text = _a1_run(W, tmp_path, "shown", LIFTED, PUSHED, content=[0.35] * 8, extra=over2, printed_too=True)
    assert "the deaf region, where the attribution read 2.0 times the measured lift" in text
    readable(text)


def test_a1_vi_judges_the_fatigue_answer_at_the_holds_line_in_force(W, tmp_path):
    """The coordinator's decision A: the lesson reads the hold's line again each window (the live bodies' answer now),
    so A1 (vi) reads each target's answer on the posture-only rows against the line in force at A1's windows. Here
    production raised the live bodies' answer: the line read 0.16 at the start and 0.25 at A1's windows, and a lifted
    body answering 0.26 answers fatigue as the live bodies do now (0.25 + 2 x 0.03 / sqrt 3 = 0.285). Fails on 67335783f
    (it judged against the start's 0.16: 0.195)."""
    A = _a1_run(W, tmp_path, "now", LIFTED, PUSHED, content=[0.35] * 8, fatigue=[0.26] * 8, fatigue_line=0.25,
                fatigue_p=[0.26] * 8)
    b1 = A["A1"]["bodies"][1]
    assert A["A1"]["state"] == "met" and b1["answers_fatigue_as_live"], A["A1"]
    assert b1["fatigue_line"] == pytest.approx(0.25 + 2 * 0.03 / 3 ** 0.5)
    tired = _a1_run(W, tmp_path, "tired_now", LIFTED, PUSHED, content=[0.35] * 8, fatigue_line=0.25,
                    fatigue_p=[0.30, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.26])
    assert tired["A1"]["missed"] == [1], "0.30 over the line in force"


def test_a1_judges_the_fatigue_answer_at_the_holds_line_the_live_bodies_largest_at_the_start(W, tmp_path):
    """The coordinator's decision (record section 45): A1 (vi) reads each target's fatigue answer against THE HOLD'S
    LINE, the one the lesson recorded in its start (the live bodies' largest answer then), plus 2 FATIGUE_SIGMA_HI /
    sqrt 3; a start that recorded none is read at the largest start answer of the judged bodies that were not targets,
    never their median. Here the live bodies answered 0.12 but body 2 answered 0.20: a lifted body at 0.19 answers
    fatigue as the live bodies do (the median, 0.12, would fail it); at 0.26 it does not."""
    fat = [0.10, 0.20, 0.05, 0.12, 0.12, 0.12, 0.12, 0.10]
    lifted = [0.19, 0.20, 0.05, 0.12, 0.12, 0.12, 0.12, 0.19]
    for name, st in (("recorded", start_block(fatigue=fat)),
                     ("unrecorded", start_block(fatigue=fat, fatigue_line=None))):
        A = _a1_run(W, tmp_path, name, LIFTED, PUSHED, start=st, fatigue=lifted)
        b1 = A["A1"]["bodies"][1]
        assert A["A1"]["live_fatigue_at_start"] == pytest.approx(0.20), name
        assert b1["answers_fatigue_as_live"] and b1["fatigue_line"] == pytest.approx(0.20 + 2 * 0.03 / 3 ** 0.5)
    own = _a1_run(W, tmp_path, "own", LIFTED, PUSHED, start=start_block(fatigue=fat, fatigue_line=0.17),
                  fatigue=lifted)
    assert own["A1"]["live_fatigue_at_start"] == pytest.approx(0.17), "the lesson's own recorded line comes first"
    tired = _a1_run(W, tmp_path, "tired26", LIFTED, PUSHED, start=start_block(fatigue=fat),
                    fatigue=[0.26, 0.20, 0.05, 0.12, 0.12, 0.12, 0.12, 0.19])
    assert tired["A1"]["missed"] == [1] and tired["A1"]["bodies"][1]["answers_fatigue_as_live"] is False


def test_the_hold_is_printed_with_its_line_and_the_bodies_it_pulls_back(W, tmp_path):
    """The hourly sweep names the fatigue hold: its line as a share (a large 300-s fatigue change moves the token that far
    relative to its movement with the posture), the bodies the lesson has pushed and the ones the hold pulled back at the
    newest report; a start that recorded no line says the hold is not acting. Version 7 (the review's finding 4; the
    coordinator's decision A), as readings: the line IN FORCE (the lesson reads it again each window) and the step it was
    read beside the start's, THE LIVE BODIES' ANSWER NOW (the judged bodies never pushed, on the posture-only rows the
    line is read on) and the share of the steps over the last 20 reports in which the hold acted. Fails on 67335783f
    (the start's line only, none of the readings)."""
    rows = [rec(N + 100 * k, held=[b == 0 for b in range(8)]) for k in range(1, 30)]
    text = printed(W, make_runs(tmp_path / "held", rows))
    assert "the fatigue hold" in text and "at most 16% as far" in text
    assert "bodies pushed so far 1, 8; pulled back at the newest report: body 1" in text
    readable(text)
    fp = [0.30, 0.17, 0.05, 0.18, 0.19, 0.18, 0.18, 0.29]
    rows = [rec(N + 100 * k, held=[b == 0 for b in range(8)], fatigue_line=(0.2 if k > 60 else 0.16),
                fatigue_line_step=(S + 6000 if k > 60 else S), fatigue_p=fp, hold_any=(0.25 if k > 40 else 0.5))
            for k in range(1, 70)]
    ev = judged(W, make_runs(tmp_path / "now", rows))
    h = ev["judgement"]["hold"]
    assert h["fatigue_line"] == pytest.approx(0.2) and h["fatigue_line_start"] == pytest.approx(0.16)
    assert h["live_answer_now"]["bodies"] == [2, 4, 5, 6, 7] and h["live_answer_now"]["median"] == pytest.approx(0.18)
    assert h["hold_share_recent"] == pytest.approx(0.25) and h["recent_reports"] == 20
    text = printed(W, make_runs(tmp_path / "now2", rows))
    assert "the line now 20%, read at step 56,100 (16% at the start)" in text
    # fix round 2 (the re-review's finding 3): the lesson's lift of the never-pushed bodies lowers their answer, and F
    # with it (9-10 in 100 at 415,075 on the lesson's path, 18 at 380,550): the printed words say the spill is in it
    assert ("the live bodies' answer now, the lesson's spill included (bodies 2, 4, 5, 6, 7, the ones never pushed) 18%, "
            "from 17% to 19%") in text
    assert "the lesson's spill into those bodies included, which lowers it and holds harder" in text
    assert "the hold acted in 25% of the steps over the last 20 reports" in text
    readable(text)
    rows = [rec(N + 100 * k, start=start_block(fatigue_line=None)) for k in range(1, 30)]
    text = printed(W, make_runs(tmp_path / "none", rows))
    assert "not acting - no live body at the start" in text
    readable(text)


def test_the_success_metric_is_the_gap_closed_toward_the_live_level_never_a_floor(W, tmp_path):
    """The coordinator: the deaf bodies' response per unit movement toward the live bodies', never a floor as the goal.
    At every window after the start: each target's share of the gap from its start to the line, pooled, and beside it
    the gap closed toward the live bodies' own level at that window."""
    half = [2.0, 3.6, 0.3, 3.6, 3.6, 3.6, 3.6, 2.1]
    ho = dict(step=S + 6000, encoder_step=S, ratio=[1.5, 3.4, 0.3, 3.4, 3.4, 3.4, 3.4, 1.65],
              content_gain_matched=[0.29, 0.3, 0.1, 0.3, 0.3, 0.3, 0.3, 0.28])
    rows = [rec(step, ratio=half, held_out=ho if step >= S + 6000 else None) for step in range(N + 100, S + 6001, 100)]
    ev = judged(W, make_runs(tmp_path / "half", rows))
    sx = ev["judgement"]["acceptance"]["success"]
    assert sx["bodies"][1]["gap_closed_to_line"] == pytest.approx(0.5)
    assert sx["held_out"]["bodies"][1]["gap_closed_to_line"] == pytest.approx(0.25), \
        "version 6: the window read through the encoder saved one window earlier"
    assert sx["held_out"]["bodies"][8]["content_gain_matched"] == pytest.approx(0.28)
    assert sx["live_level_at_start"] == pytest.approx(3.5)
    assert sx["bodies"][8]["gap_closed_to_line"] == pytest.approx((2.1 - 1.2) / (3.0 - 1.2))
    assert sx["live_level_now"] == pytest.approx(3.6)
    assert sx["pooled_to_live_now"] == pytest.approx((2.05 - 1.1) / (3.6 - 1.1))
    text = printed(W, make_runs(tmp_path / "shown", rows))
    assert "toward the live level" in text and "body 1 50%" in text
    assert "held out (the window read through the encoder saved at step 50,100" in text and "body 1 25%" in text
    assert "unmeasured until the prediction-dependence job" in text
    readable(text)


def test_a1_and_a2_with_no_targets_say_so_and_never_pass(W, tmp_path):
    none_deaf = start_block(ratio=[3.5] * 8)
    A = _a1_run(W, tmp_path, "none", [3.5] * 8, [0.0] * 8, start=none_deaf)
    assert A["A1"]["state"] == "no targets" and A["A1_targets"] == []
    text = printed(W, make_runs(tmp_path / "shown", [rec(N + 100 * k, ratio=[3.5] * 8, start=none_deaf)
                                                     for k in range(1, 70)]))
    assert "none: no body was deaf at the start" in text and "no body was deaf at the start" in text
    readable(text)


# ------------------------------------------------------------------ A2 on the ratio
A2_FROM = S + 10000 + 6000                # the first horizon checkpoint: start + the horizon floor + one window
PRE = [3.6] * 8                           # the calibration's ratios of 344,400 in these fixtures


def _a2_extra(**kw):
    """The lines file's conversion (a measured log gap of 0.02: the real rows read exp(0.02) of the rebuilt ones), the
    classes at the switch and the pre-deploy state."""
    doc = dict(gap_by_body=[0.02] * 8, gap_by_body_source=["measured on this body"] * 8,
               deaf_at_switch=[True, False, False, False, False, False, False, True],
               pre_deploy=dict(step=344400, ratio=list(PRE)))
    doc.update(kw)
    return doc


def _a2_run(W, tmp_path, name, ratio, step=A2_FROM, horizon_gap=None, start=None, extra=None, n=25, N_=N):
    st = start_block() if start is None else start
    deploy = make_runs(tmp_path / name, [rec(N + 100 * (k + 1), start=st) for k in range(n)],
                       lines_extra=_a2_extra(**(extra or {})))
    if ratio is not None:
        (deploy / "proprio_repair_a2.json").write_text(json.dumps(dict(
            version="proprio_token_repair_v5", N=N_, step=step, ratio=ratio, gap_by_body=horizon_gap)),
            encoding="utf-8")
    return deploy


def test_a2_is_judged_at_the_deaf_live_line_on_the_horizon_checkpoint_converted_by_the_log_gap(W, tmp_path):
    """A2 (version 6; the operator's decision, record section 38, option 1): on the horizon checkpoint, by the
    calibration's ratio converted by the measured log gap, every A1 target at or above DEAF x exp(-2 SIGMA_LOG) (2.0 x
    0.819 = 1.64 here; 2.26 on the committed readings) and no other judged body's ln ratio more than K6_FALL_LOG under
    its switch reading. Controls: without the conversion the same readings miss (the gap decides bodies 1 and 8 here); a
    target under the line misses; another body falling past the line misses; body 3 never counts; a target at 2.40,
    between the deaf/live line and the live level's line (2.46), is met (version 5 judged it at the live level and
    missed it). The live level stays the goal: each target's share of the gap from its start to LINE is printed beside
    it."""
    held = [1.62, 3.4, 0.01, 3.4, 3.4, 3.4, 3.4, 1.63]
    ev = judged(W, _a2_run(W, tmp_path, "met", held))
    A2 = ev["judgement"]["acceptance"]["A2"]
    assert A2["state"] == "met", A2
    assert A2["line"] == pytest.approx(2.0 * math.exp(-0.2)) and A2["targets"] == [1, 8] and A2["step"] == A2_FROM
    assert A2["bodies"][1]["reading"] == pytest.approx(1.62 * math.exp(0.02)) and A2["bodies"][1]["reached"]
    assert A2["bodies"][1]["gap_closed_to_live_level"] == pytest.approx((1.62 * math.exp(0.02) - 1.0) / (3.0 - 1.0))
    assert A2["bodies"][4]["fall_log"] == pytest.approx(math.log(3.5 / 3.4)) and A2["bodies"][4]["kept"]
    assert 3 not in A2["bodies"] and A2["conversion"] == "the switch's measured gap"
    raw = judged(W, _a2_run(W, tmp_path, "raw", held, extra=dict(gap_by_body=None)))["judgement"]["acceptance"]["A2"]
    assert raw["state"] == "missed" and raw["missed"] == [1, 8] and raw["conversion"].startswith("none")
    own = judged(W, _a2_run(W, tmp_path, "own", held, horizon_gap=[0.0] * 8))["judgement"]["acceptance"]["A2"]
    assert own["missed"] == [1, 8] and own["conversion"] == "the horizon checkpoint's own measured gap"
    low = list(held)
    low[0] = 1.5
    got = judged(W, _a2_run(W, tmp_path, "low", low))["judgement"]["acceptance"]["A2"]
    assert got["state"] == "missed" and got["missed"] == [1] and not got["lost"]
    fell = list(held)
    fell[3] = 2.3                                           # body 4: ln(3.5 x 1.02 / (2.3 x 1.02)) = 0.42 > 0.33
    got = judged(W, _a2_run(W, tmp_path, "fell", fell))["judgement"]["acceptance"]["A2"]
    assert got["state"] == "missed" and got["lost"] == [4]
    between = [2.40, 3.4, 0.01, 3.4, 3.4, 3.4, 3.4, 2.40]
    got = judged(W, _a2_run(W, tmp_path, "between", between, extra=dict(gap_by_body=None)))
    assert got["judgement"]["acceptance"]["A2"]["state"] == "met", \
        "2.40 lies over the deaf/live line (1.64) and under the live level's (3.0 x 0.819 = 2.46)"
    text = printed(W, _a2_run(W, tmp_path, "printed", held))
    assert "A2 (never a stop; restated at the deaf/live line)" in text and "checkpoint 66,100" in text
    assert ": met" in text and "the live level, 3.00, stays the goal" in text
    readable(text)


def test_a2_prints_the_generalization_reading_at_each_targets_heading(W, tmp_path):
    """The third review's SEMANTIC: the lift of bodies 2 and 6 did not carry to other movement at their heading (8 in
    100 of the log gap). The A2 file carries the calibration's heading reading at the horizon, the lines file the
    switch's; A2 prints, per target, how far the other bodies' movement at its heading moved from the switch toward the
    live level: a reading, never a pass line."""
    held = [1.62, 3.4, 0.01, 3.4, 3.4, 3.4, 3.4, 1.63]
    was = dict(at_body={"1": dict(median_ratio=0.9), "8": dict(median_ratio=0.9)})
    now = dict(at_body={"1": dict(median_ratio=1.95), "8": dict(median_ratio=0.9)})
    deploy = _a2_run(W, tmp_path, "gen", held, extra=dict(switch_heading_transfer=was))
    a2 = json.loads((deploy / "proprio_repair_a2.json").read_text(encoding="utf-8"))
    (deploy / "proprio_repair_a2.json").write_text(json.dumps(dict(a2, heading_transfer=now)), encoding="utf-8")
    A2 = judged(W, deploy)["judgement"]["acceptance"]["A2"]
    assert A2["state"] == "met", "the reading never decides"
    assert A2["generalization"][1]["gap_closed_to_live_level"] == pytest.approx(0.5)
    assert A2["generalization"][8]["gap_closed_to_live_level"] == pytest.approx(0.0)
    text = printed(W, deploy)
    assert "body 1's heading 50% of the way from the switch to the live level" in text
    readable(text)


def test_the_distance_from_the_pre_deploy_state_is_a_share_reported_at_every_judgment_and_never_decides(W, tmp_path):
    held = [1.62, 3.4, 0.01, 3.4, 3.4, 3.4, 3.4, 1.63]
    ev = judged(W, _a2_run(W, tmp_path, "far", held))
    A = ev["judgement"]["acceptance"]
    js = A["from_pre_deploy"]["judgments"]
    assert [j["judgment"] for j in js] == ["the lesson's start", "A2"]
    assert js[0]["bodies"][1] == pytest.approx(1.0 / (3.6 * math.exp(0.02))) and 3 not in js[0]["bodies"]
    assert js[-1]["bodies"][1] == pytest.approx(1.62 / 3.6) and js[-1]["step"] == A2_FROM
    level = judged(W, _a2_run(W, tmp_path, "level", held, extra=dict(pre_deploy=dict(step=344400, ratio=held))))
    assert level["judgement"]["acceptance"]["A2"]["state"] == A["A2"]["state"] == "met", "the distance never decides"
    text = printed(W, _a2_run(W, tmp_path, "shown", held))
    assert "distance from the pre-deploy state" in text and "344,400" in text and "body 1 45%" in text
    readable(text)
    ev = judged(W, make_runs(tmp_path / "none", after_reports()))
    assert ev["judgement"]["acceptance"]["from_pre_deploy"]["judgments"] == []
    assert "no pre-deploy state" in ev["judgement"]["acceptance"]["from_pre_deploy"]["why"]


def test_a2_waits_for_its_file_and_refuses_one_read_before_the_horizon(W, tmp_path):
    held = [1.62, 3.4, 0.01, 3.4, 3.4, 3.4, 3.4, 1.63]
    got = judged(W, _a2_run(W, tmp_path, "none", None))["judgement"]["acceptance"]["A2"]
    assert got["state"] == "waiting" and "step 66,100" in got["why"] and got["checkpoint_from"] == A2_FROM
    got = judged(W, _a2_run(W, tmp_path, "early", held, step=A2_FROM - 100))["judgement"]["acceptance"]["A2"]
    assert got["state"] == "refused" and "before step 66,100" in got["why"]
    ev = judged(W, _a2_run(W, tmp_path, "foreign", held, N_=N + 5))
    assert ev["judgement"]["acceptance"]["A2"]["state"] == "waiting"
    assert "is about step 50005" in ev["judgement"]["acceptance"]["A2"]["why"]
    none_deaf = start_block(ratio=[3.5] * 8)
    A = judged(W, _a2_run(W, tmp_path, "over", held, start=none_deaf))["judgement"]["acceptance"]
    assert A["A2"]["state"] == "no targets" and A["A2"]["targets"] == [], "a loss is still judged with no targets"


def _cal_v6(tmp_path, rows, name="cal.json", **rules):
    doc = dict(rules=dict(dict(version=6, LINE=3.75, DEAF=2.86, SKIP=[3], K2_share_diff=0.3,
                               window_noise=dict(sigma=0.117, sigma_hi=0.171, k6_fall=0.386),
                               content_noise=dict(sigma=0.044, sigma_hi=0.069),
                               fatigue_noise=dict(sigma=0.02, sigma_hi=0.03),
                               post_levels=dict(live_median=4.04)), **rules), rows=rows)
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _gap_file(tmp_path, name, body, real, rebuilt, share_real=0.27, share_rebuilt=0.30):
    s_real, s_reb = 0.3, 0.3
    doc = dict(lived=dict(body=body), readings={"200.0": dict(
        real=dict(ratio=real, spread=s_real, jitter=2 * share_real * s_real ** 2),
        rebuilt=dict(ratio=rebuilt, spread=s_reb, jitter=2 * share_rebuilt * s_reb ** 2))})
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_the_lines_file_is_built_from_version_6s_pause_results(W, tmp_path):
    """LINE and DEAF in the lesson's units: the calibrated values times exp of the mean measured log gap (body 4 of
    415,075 read 3.51 on the real rows and 3.48 on the rebuilt ones), rounded down; each body's conversion; the switch's
    readings (the ratio, the jitter share difference, the fatigue answer, the heading reading); the pre-deploy state;
    A1's horizon from rule LS. Refused: a version-5 calibration, one that set no K2 difference line or no fatigue
    noise, one that did not read the switch."""
    switch = dict(loaded_step=N, ratio=dict(ratio=[1.29, 1.08, 0.32, 4.06, 4.25, 1.08, 4.36, 1.28],
                                            jitter=[0.001] * 8, spread=[0.1] * 8, content=[0.24] * 8,
                                            jitter_share=[0.4] * 8, posture_jitter_share=[0.35] * 8,
                                            fatigue_answer=[0.1] * 8,
                                            heading_transfer=dict(at_body={"1": dict(median_ratio=0.9)})))
    pre = dict(loaded_step=W.PRE_DEPLOY_STEP, ratio=dict(ratio=[3.63, 3.63, 0.43, 4.02, 4.40, 3.21, 3.48, 3.44]))
    cal = _cal_v6(tmp_path, [pre, switch])
    meas = tmp_path / "meas.json"
    meas.write_text(json.dumps(dict(rules=dict(LS=dict(checkpoints=[dict(step=N - 7, A1_horizon=1800.0),
                                                                     dict(step=N - 3, A1_horizon=2400.0)])))),
                    encoding="utf-8")
    g4 = _gap_file(tmp_path, "gap_b4.json", 3, 3.512, 3.479)
    g1 = _gap_file(tmp_path, "gap_b1.json", 0, 1.30, 1.27)
    got = W.build_lines(N, cal, meas, [g4, g1])
    mean_gap = (math.log(3.512 / 3.479) + math.log(1.30 / 1.27)) / 2
    assert got["LINE"] == math.floor(100 * 3.75 * math.exp(mean_gap)) / 100 and got["line_rebuilt"] == 3.75
    assert got["DEAF"] == math.floor(100 * 2.86 * math.exp(mean_gap)) / 100 and got["SKIP"] == [3]
    assert got["gap_by_body"][3] == pytest.approx(math.log(3.512 / 3.479)), "a measured body keeps its own gap"
    assert got["gap_by_body"][7] == pytest.approx(math.log(1.30 / 1.27)), "an unmeasured deaf body takes the deaf mean"
    assert got["deaf_at_switch"] == [True, True, True, False, False, True, False, True]
    assert got["pre_deploy"] == dict(step=344400, ratio=pre["ratio"]["ratio"])
    assert got["switch_ratio"] == switch["ratio"]["ratio"] and got["switch_share_diff"] == pytest.approx([0.05] * 8)
    assert got["switch_fatigue"] == [0.1] * 8 and got["fatigue_sigma_hi"] == 0.03 and got["K2_share_diff"] == 0.3
    assert got["switch_heading_transfer"]["at_body"]["1"]["median_ratio"] == 0.9
    assert got["share_ratio_gap"] == pytest.approx(0.9) and got["A1_horizon"] == 2400.0
    assert got["window_sigma_log"] == 0.117 and got["k6_fall_log"] == 0.386 and got["content_sigma_hi"] == 0.069
    plain = W.build_lines(N, cal)
    assert plain["LINE"] == 3.75 and "unmeasured" in plain["input_gap"] and plain["gap_by_body"] is None
    with pytest.raises(ValueError, match="not version 6"):
        W.build_lines(N, _cal_v6(tmp_path, [switch], "v5.json", version=5))
    with pytest.raises(ValueError, match="sets no K2_share_diff"):
        W.build_lines(N, _cal_v6(tmp_path, [switch], "nok2.json", K2_share_diff=None))
    with pytest.raises(ValueError, match="sets no fatigue_noise"):
        W.build_lines(N, _cal_v6(tmp_path, [switch], "nofat.json", fatigue_noise=None))
    with pytest.raises(ValueError, match="no reading of the switch's checkpoint"):
        W.build_lines(N + 1, cal)
    bad = tmp_path / "gap_bad.json"
    bad.write_text(json.dumps(dict(lived=dict(body=0), readings={"200.0": dict(real=dict(ratio=0.0))})),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="no real and rebuilt ratio"):
        W.build_lines(N, cal, None, [bad])
    out = tmp_path / "lines" / "proprio_repair_lines.json"
    assert W.main(["lines", "--step", str(N), "--calibrate", str(cal), "--measure", str(meas), "--gap", str(g4),
                   str(g1), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["A1_horizon"] == 2400.0
    assert W.main(["lines", "--step", str(N + 1), "--calibrate", str(cal), "--out", str(tmp_path / "no.json")]) == 1
    assert not (tmp_path / "no.json").exists()
    assert plain["attribution_over_read"] is None, "no version-7 pairs: A1 (ii) is a reading"


def test_the_lines_file_carries_the_attributions_over_read_per_class_from_version_7s_pairs(W, tmp_path):
    """The review's finding 5 and the coordinator's decision F: the lines file carries, per class, the largest over rule
    LS version 7's targets of the attribution's bound (the mean attribution over the paired gain's lower bound); the live
    region is a target at or over DEAF x exp(-2 SIGMA_LOG) on the calibration's scale (2.86 x exp(-0.234) = 2.26), the
    deaf region one under it; a target the pairs did not bound makes its class unbounded. FIX ROUND 2 (the re-review's
    findings 2 and 4, the coordinator's decision): the divisors come from THE SWITCH CHECKPOINT'S OWN PAIRS only (the
    measure result of step N; another checkpoint's targets are kept as readings, never a divisor: 380,550's deaf class
    gave a divisor that fails a working lesson), ONE MEASURE RESULT PER CHECKPOINT (a stage-1 file beside its stage-2
    file is refused), and the file carries A1 (ii)'s band between the classes (1.29-2.69 on the calibration's scale,
    converted by the mean input gap as DEAF is). Fails on ee2e66f10 (the largest over every result given: 1.9 from
    380,550 here; two results for one checkpoint taken together; no band)."""
    switch = dict(loaded_step=N, ratio=dict(ratio=[1.29, 1.08, 0.32, 4.06, 4.25, 1.08, 4.36, 1.28],
                                            jitter=[0.001] * 8, spread=[0.1] * 8, jitter_share=[0.4] * 8,
                                            posture_jitter_share=[0.35] * 8, fatigue_answer=[0.1] * 8))
    cal = _cal_v6(tmp_path, [switch])

    def meas(name, bodies, version=7):
        p = tmp_path / name
        p.write_text(json.dumps(dict(rules=dict(LS=dict(version=version, checkpoints=[
            dict(step=s, A1_horizon=h, bodies=[dict(body=b, judged=True, target=True, start=st, over_read_bound=o,
                                                    state="lifted") for b, st, o in bs])
            for s, h, bs in bodies])))), encoding="utf-8")
        return p
    a = meas("m380550.json", [(380550, 2200.0, [(1, 0.98, 2.9), (4, 2.69, 9.4)])])
    b = meas("mN.json", [(N, 3000.0, [(1, 1.29, 1.55), (2, 1.08, 1.47), (4, 2.69, 3.4)])])
    got = W.build_lines(N, cal, [a, b])
    over = got["attribution_over_read"]
    assert over["classes"] == {"deaf region": 1.55, "live region": 3.4}, "the switch checkpoint's own pairs only"
    assert over["from_step"] == N and got["A1_horizon"] == 3000.0, "the horizon still reads every checkpoint"
    assert over["boundary_rebuilt"] == pytest.approx(2.86 * math.exp(-2 * 0.117))
    assert len(over["bodies"]) == 3 and {x["region"] for x in over["bodies"]} == {"deaf region", "live region"}
    assert over["class_starts_rebuilt"] == {"deaf region": [1.08, 1.29], "live region": [2.69, 2.69]}
    assert {(x["step"], x["over_read_bound"]) for x in over["other_checkpoints"]} == {(380550, 2.9), (380550, 9.4)}, \
        "380,550's targets are readings, never a divisor"
    assert over["band_rebuilt"] == [1.29, 2.69] and over["band"] == [1.29, 2.69], "no gap measured: unconverted"
    g4 = _gap_file(tmp_path, "gap_b4.json", 3, 3.512, 3.479)
    conv = W.build_lines(N, cal, [a, b], [g4])["attribution_over_read"]
    assert conv["band"] == pytest.approx([1.29 * 3.512 / 3.479, 2.69 * 3.512 / 3.479], abs=1e-4)
    # one result per checkpoint: a checkpoint's stage-1 and stage-2 results together are refused
    b2 = meas("mN_stage2.json", [(N, 3000.0, [(1, 1.29, 1.3)])])
    with pytest.raises(ValueError, match="one result per checkpoint"):
        W.build_lines(N, cal, [a, b, b2])
    # no result read the switch's checkpoint: no divisor, A1 (ii) a reading, and the file says why
    none_here = W.build_lines(N, cal, [a])["attribution_over_read"]
    assert none_here["classes"] == {} and "reading only" in none_here["why"] and len(none_here["other_checkpoints"]) == 2
    c = meas("m_unb.json", [(N, 3000.0, [(1, 1.29, None)])])
    assert W.build_lines(N, cal, [a, c])["attribution_over_read"]["classes"]["deaf region"] == "unbounded"
    v6 = meas("m_v6.json", [(N, 3000.0, [(1, 1.29, 1.4)])], version=None)
    assert W.build_lines(N, cal, [v6])["attribution_over_read"] is None, "version 6's rule measured no bound"
    out = tmp_path / "lines" / "l.json"
    assert W.main(["lines", "--step", str(N), "--calibrate", str(cal), "--measure", str(a), str(b), "--out",
                   str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["attribution_over_read"]["classes"]["live region"] == 3.4
    assert W.main(["lines", "--step", str(N), "--calibrate", str(cal), "--measure", str(b), str(b2), "--out",
                   str(tmp_path / "two.json")]) == 1 and not (tmp_path / "two.json").exists()


def test_a1_ii_takes_its_divisor_from_the_switch_checkpoints_own_pairs_and_a_band_target_the_larger_classs(W, tmp_path):
    """The re-review's findings 2 and 4 (fix round 2, the coordinator's decision). A body the lesson lifts to LINE and
    releases carries an attribution of about its log shortfall times production's own over-read o_p, so (ii) passes when
    o_p >= k / 2. Here the attribution equals the shortfall (o_p = 1.00, the lesson's own path): with the deaf class's
    divisor from the switch checkpoint's pairs (k = 1.55, 415,075's deaf targets on version 6's numbers) it passes; with
    the divisor the fix round took from 380,550's deaf class (k = 2.93 at the widest spread) the same working lesson
    fails. A target starting in the band between the classes takes the larger class's divisor (here the live class's,
    3.3), and is a reading only when either class is unmeasured at the switch's checkpoint. Fails on ee2e66f10 (no
    band: a target at 1.6, under the class boundary 1.64, took the deaf class's divisor)."""
    short1, short8 = math.log(3.0 / 1.0), math.log(3.0 / 1.2)
    working = [short1, 0, 0, 0, 0, 0, 0, short8]                 # attribution = shortfall: o_p = 1.00
    own = dict(attribution_over_read=dict(classes={"deaf region": 1.55, "live region": 3.3}, band=[1.5, 1.9],
                                          from_step=N))
    A, text = _a1_run(W, tmp_path, "own", LIFTED, working, content=[0.35] * 8, extra=own, printed_too=True)
    assert A["A1"]["state"] == "met" and A["A1"]["bodies"][1]["over_read"] == 1.55, A["A1"]
    assert A["A1"]["bodies"][1]["attributed_corrected"] == pytest.approx(short1 / 1.55)
    assert "measured at the switch's checkpoint (step 50,000)" in text
    assert "a body starting in the band between the classes (a response from 1.50 to 1.90 times" in text
    readable(text)
    old = dict(attribution_over_read=dict(classes={"deaf region": 2.93, "live region": 9.4}))
    A = _a1_run(W, tmp_path, "old", LIFTED, working, content=[0.35] * 8, extra=old)
    assert A["A1"]["state"] == "missed" and A["A1"]["missed"] == [1, 8], "380,550's divisor fails a working lesson"
    # a target starting in the band (1.6: under the class boundary 2.0 x exp(-0.2) = 1.64) takes the larger divisor
    band_start = start_block(ratio=[1.6, 3.5, 0.3, 3.5, 3.5, 3.5, 3.5, 1.2])
    lift = [0.8, 0, 0, 0, 0, 0, 0, short8]                        # 0.8 / 1.55 = 0.52 passes, 0.8 / 3.3 = 0.24 does not
    A = _a1_run(W, tmp_path, "band", LIFTED, lift, content=[0.35] * 8, start=band_start, extra=own)
    b1 = A["A1"]["bodies"][1]
    assert b1["attributed_class"] == "band between the classes" and b1["over_read"] == 3.3
    assert b1["lifted_by_lesson"] is False and A["A1"]["missed"] == [1], "half of ln(3.0 / 1.6) = 0.31 over 0.24"
    assert A["A1"]["bodies"][8]["attributed_class"] == "deaf region", "1.2 lies under the band"
    unb = dict(attribution_over_read=dict(classes={"deaf region": 1.55, "live region": "unbounded"}, band=[1.5, 1.9]))
    A = _a1_run(W, tmp_path, "band_unb", LIFTED, lift, content=[0.35] * 8, start=band_start, extra=unb)
    assert A["A1"]["bodies"][1]["over_read"] == "unbounded" and A["A1"]["missed"] == [1]
    # the live class unmeasured at the switch's checkpoint (415,075 has no live-region target): a band target's (ii)
    # is a reading, deciding nothing
    deaf_only = dict(attribution_over_read=dict(classes={"deaf region": 1.55}, band=[1.5, 1.9], from_step=N))
    A, text = _a1_run(W, tmp_path, "band_read", LIFTED, lift, content=[0.35] * 8, start=band_start, extra=deaf_only,
                      printed_too=True)
    b1 = A["A1"]["bodies"][1]
    assert b1["lifted_by_lesson_judged"] is False and b1["lifted_by_lesson"] is None and A["A1"]["state"] == "met"
    assert "band between the classes" in b1["attributed_reading"]
    assert "the live region, unmeasured there, so (ii) is a reading for it" in text
    readable(text)


def test_a2_prints_the_planners_goal_pick_as_a_reading(W, tmp_path):
    """The review's finding 7 and the coordinator's decision: the lift changes the input to the planner's goal pick
    (section 44: a shuffled token changed 19 of 126 picks at the switch) and no line watched it. The A2 file carries
    credit.py's goal-pick result on the horizon checkpoint (`a2 --credit`, with --compare-encoder on the switch's
    checkpoint): the share of goal picks that change when the proprioception token is read through the switch's encoder
    instead of the horizon's, beside the shuffle's; A2 prints it and it never decides. A result of another checkpoint is
    refused. Fails on 67335783f (no goal-pick reading)."""
    cal = tmp_path / "cal_h.json"
    cal.write_text(json.dumps(dict(rules=dict(version=6), rows=[dict(loaded_step=A2_FROM + 200, ratio=dict(
        ratio=[3.0] * 8, content=[0.4] * 8, posture=[0.07] * 8))])), encoding="utf-8")
    lines_ = dict(N=N, deaf_at_switch=[True] + [False] * 6 + [True], gap_by_body=[0.02] * 8)
    credit = tmp_path / "credit.json"
    credit.write_text(json.dumps(dict(step=A2_FROM + 200, loaded_step=A2_FROM + 200,
                                      compare_encoder="x/checkpoints/ckpt_step_%d" % N, compare_encoder_step=N,
                                      goal_pick=dict(offset=0, events=18,
                                                     switch_encoder=dict(picks_changed=30, picks=126, per_100=23.8),
                                                     shuffle_1=dict(picks_changed=40, picks=126, per_100=31.7),
                                                     mean=dict(picks_changed=0, picks=126, per_100=0.0)))),
                      encoding="utf-8")
    got = W.build_a2(N, A2_FROM + 200, cal, lines_, credit=credit)
    gp = got["goal_pick"]
    assert gp["compared_with_the_switch"] and gp["arms"]["switch_encoder"]["picks_changed"] == 30
    assert set(gp["arms"]) == {"switch_encoder", "shuffle_1", "mean"}
    with pytest.raises(ValueError, match="not the horizon checkpoint"):
        W.goal_pick_of(credit, A2_FROM + 300, N)
    held = [1.62, 3.4, 0.01, 3.4, 3.4, 3.4, 3.4, 1.63]
    deploy = _a2_run(W, tmp_path, "goal", held)
    a2 = json.loads((deploy / "proprio_repair_a2.json").read_text(encoding="utf-8"))
    (deploy / "proprio_repair_a2.json").write_text(json.dumps(dict(a2, goal_pick=gp)), encoding="utf-8")
    A2 = judged(W, deploy)["judgement"]["acceptance"]["A2"]
    assert A2["state"] == "met" and A2["goal_pick"]["arms"]["shuffle_1"]["picks"] == 126, "a reading, never a line"
    text = printed(W, deploy)
    assert "the planner's goal pick (a reading, never a line): 30 of 126 goal picks (24%) change at the horizon" in text
    assert "a shuffled token changes 40 of 126 goal picks (32%) there, where it changed 19 of 126 (15%) at the switch" \
        in text
    readable(text)
    none = judged(W, _a2_run(W, tmp_path, "nogoal", held))["judgement"]["acceptance"]["A2"]
    assert none["state"] == "met" and "unmeasured" in none["goal_pick"]


def test_the_a2_file_is_built_from_the_horizon_checkpoints_calibration(W, tmp_path):
    cal = tmp_path / "cal_h.json"
    cal.write_text(json.dumps(dict(rules=dict(version=6, note="no pre-deploy checkpoint read"),
                                   rows=[dict(loaded_step=A2_FROM + 200, ratio=dict(
                                       ratio=[3.0] * 8, content=[0.4] * 8, posture=[0.07] * 8,
                                       heading_transfer=dict(at_body={"1": dict(median_ratio=2.0)})))])),
                   encoding="utf-8")
    lines_ = dict(N=N, deaf_at_switch=[True] + [False] * 6 + [True], gap_by_body=[0.02] * 8)
    got = W.build_a2(N, A2_FROM + 200, cal, lines_)
    assert got["step"] == A2_FROM + 200 and got["ratio"] == [3.0] * 8 and got["gap_by_body"] is None
    assert got["heading_transfer"]["at_body"]["1"]["median_ratio"] == 2.0, "the generalization reading rides the file"
    gap = _gap_file(tmp_path, "gap_h.json", 0, 1.3, 1.27)
    got = W.build_a2(N, A2_FROM + 200, cal, lines_, [gap])
    assert got["gap_by_body"] == pytest.approx([math.log(1.3 / 1.27)] * 8)
    assert got["gap_by_body_source"][0] == "measured on this body"
    assert got["gap_by_body_source"][7] == "the mean of the measured deaf bodies"
    with pytest.raises(ValueError, match="no reading of checkpoint"):
        W.build_a2(N, A2_FROM, cal, lines_)
    with pytest.raises(ValueError, match="not about the switch"):
        W.build_a2(N + 1, A2_FROM + 200, cal, lines_)
    lf = tmp_path / "lines.json"
    lf.write_text(json.dumps(lines_), encoding="utf-8")
    out = tmp_path / "a2" / "proprio_repair_a2.json"
    assert W.main(["a2", "--step", str(N), "--checkpoint-step", str(A2_FROM + 200), "--calibrate", str(cal),
                   "--lines", str(lf), "--gap", str(gap), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["N"] == N
    assert W.main(["a2", "--step", str(N), "--checkpoint-step", "1", "--calibrate", str(cal), "--lines", str(lf),
                   "--out", str(tmp_path / "no.json")]) == 1


def test_a_missing_lines_file_makes_its_lines_wait(W, tmp_path):
    deploy = make_runs(tmp_path, after_reports(), lines=False)
    ev = judged(W, deploy)
    assert ev["lines_error"] and "no lines file" in ev["lines_error"]
    assert ev["judgement"]["lines"]["K6"]["state"] == "waiting"
    assert ev["judgement"]["lines"]["K2"]["diff_line_missing"] is True
    assert "PROPRIOCEPTION REPAIR LINES MISSING" in printed(W, deploy)
    other = tmp_path / "committed.json"
    other.write_text(json.dumps(dict(N=N + 5, LINE=3.0)), encoding="utf-8")
    ev = W.evaluate(deploy, W.StatsReader(), lines_file=other)
    assert "is about step 50005" in ev["lines_error"]
    other.write_text(json.dumps(dict(LINES_DOC, A1_horizon=None)), encoding="utf-8")
    ev = W.evaluate(deploy, W.StatsReader(), lines_file=other)
    assert ev["lines_error"] is None and "window" in ev["judgement"]["lines"]["K6"]["why"], "it waits for a window"
    text = "\n".join(W.lines(deploy, W.StatsReader(), lines_file=other))
    assert "rule LS measured no horizon: the floor of 10,000 steps" in text
    readable(text)


def test_a_taken_stop_is_not_repeated(W, tmp_path):
    rows = after_reports(3) + [rec(N + 400, nan=1)] + [rec(N + 100 * k) for k in range(5, 26)]
    deploy = make_runs(tmp_path, rows)
    why = W.breach(deploy, W.StatsReader(), lines_file=deploy / "none.json")
    assert why and "non-finite" in why
    (deploy / "stop_reason.json").write_text(json.dumps(dict(reason=why)), encoding="utf-8")
    assert W.breach(deploy, W.StatsReader(), lines_file=deploy / "none.json") is None


def test_the_hourly_watchers_five_minute_check_writes_the_stop(tmp_path, monkeypatch, capsys):
    H = importlib.import_module("test_h84_hourly_watch")
    H.healthy_resources(monkeypatch)
    monkeypatch.setenv("UEMA_WATCH_SAVED", str(tmp_path / "_saved"))
    monkeypatch.setenv("UEMA_WATCH_ARCHIVE", str(tmp_path / "_archive"))
    healthy = H.row()
    rows = [rec(N + 100 * k, base=healthy, nan=(1 if k == 4 else 0)) for k in range(1, 26)]
    deploy = make_runs(tmp_path, rows)
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(deploy), "3"])
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(time, "time", lambda: 1e9 + N + 3000)
    runpy.run_path(str(WATCHER), run_name="__main__")
    receipt = json.loads((deploy / "stop_reason.json").read_text(encoding="utf-8"))
    assert receipt["reason"].startswith("proprioception token repair: a non-finite training update")
    assert receipt["resume_policy"] == "repair_required"
    assert json.loads((deploy / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    assert not (tmp_path / "_saved" / "supervisor_control.json").exists(), "a stale watcher never pauses another run"
    state = json.loads((deploy / "proprio_repair_watch.json").read_text(encoding="utf-8"))
    assert state["crossings"][0]["line"] == "K4"
    assert "STOP RULE REACHED: proprioception token repair" in capsys.readouterr().out


def test_a_run_without_the_switch_is_never_stopped_by_the_repair(tmp_path, monkeypatch):
    H = importlib.import_module("test_h84_hourly_watch")
    assert H.watch(tmp_path, monkeypatch, baseline=H.BASE) is None
    assert not (tmp_path / "proprio_repair_watch.json").exists()


@pytest.mark.parametrize("start", [1.29, 1.290001, 1.30, 1.34, 1.60, 2.45, 2.65, 2.689999, 2.69])
@pytest.mark.parametrize("strong", [3.3, None, "unbounded"])
def test_attribution_uses_full_unmeasured_gap_but_keeps_measured_endpoints(W, start, strong):
    """Replay the actual selection branch with values from review finding 4.

    The complete acceptance path is covered by the adjacent _a1_run tests. This
    focused fixture includes values above that fixture's artificial deaf line.
    """
    import ast
    tree = ast.parse(Path(W.__file__).read_text(encoding="utf-8"))
    branches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        for i, statement in enumerate(node.body):
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cls" for t in statement.targets):
                for j in range(i + 1, len(node.body)):
                    s = node.body[j]
                    if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ii_judged" for t in s.targets):
                        branches.append(node.body[i:j + 1])
                        break
    assert len(branches) == 1
    classes = {"deaf region": 1.55}
    if strong is not None:
        classes["live region"] = strong
    scope = dict(r0=[start], b=0, boundary=2.86 * math.exp(-2 * .117),
                 band=list(W.LINES["a1ii_band"]), classes=classes, over={"from_step": 415075}, _num=W._num)
    exec(compile(ast.Module(body=branches[0], type_ignores=[]), W.__file__, "exec"), scope)
    if start == 1.29:
        assert scope["cls"] == "deaf region" and scope["k"] == 1.55 and scope["ii_judged"]
    elif start == 2.69:
        assert scope["cls"] == "live region" and scope["k"] == strong
    else:
        assert scope["cls"] == "band between the classes" and scope["k"] == strong
    if start != 1.29:
        assert scope["ii_judged"] == (strong is not None)


def test_the_senses_go_by_the_registry_s_names_with_the_explainer_s_meanings(W):
    """The watcher's printed lines name each sense as the model registry does (until 2026-10-03 they said "touch",
    "hearing" and "balance"), and the plain meaning a stop reason prints beside a name is the opening of the
    explainer's own definition (mimo_bridge/explainer/schema.py SENSE_DEFINITIONS)."""
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    assert W.SENSE_WORDS == {s: s for s in W.OTHER_SENSES + (W.BALANCE,)}
    for sense, meaning in W.SENSE_MEANS.items():
        assert SENSE_DEFINITIONS[sense].startswith(meaning), (sense, meaning)
    assert W._named("tactile") == "tactile (the skin's sense of touch)"
