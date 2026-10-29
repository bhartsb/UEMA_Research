# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The Development page's plain framing, its dated copies and their comparison (operator, 2026-09-30).

  * each stage's "passing needs about N" inverts that stage's own test at its pass line: at N the test
    passes, one fewer it does not;
  * the line a failing row must reach is the Benjamini-Hochberg line its family would draw with it among
    the discoveries (tracker.pass_line_for agrees with fdr_line);
  * the daemon's dated copy every 4 hours, the 5-day deletion that touches only its own files, and a
    missing history disk that is a logged warning, never a failure;
  * comparing two copies: stages gained and lost, counts up and down, and the flag across a change of
    counting rules.
"""
import json
import os
import time

import numpy as np
import pytest

from mimo_bridge.tools.observer import history as HI
from mimo_bridge.tools.observer import tracker as TR
from mimo_bridge.tools.observer import view as V


# ------------------------------------------------------------------ the inversions
def test_pass_line_for_agrees_with_the_family_line():
    rng = np.random.default_rng(3)
    for _ in range(300):
        others = list(rng.uniform(0, 0.2, size=int(rng.integers(0, 25))) ** 2)
        line = TR.pass_line_for(others)
        for p in (line, line * 0.999, line * 1.001 + 1e-12):
            passes = p <= TR.fdr_line(others + [p])
            assert passes == (p <= line), (others, p, line)


def _freq_p(n, ratio, b_occ, a, b_acts, phi, phi_b, chances=1):
    return min(1.0, chances * max(TR.two_sample_above(n, ratio, b_occ, 1.0),
                                  TR.two_sample_above_dispersed(n * a, ratio, phi, b_acts, 1.0, phi_b)))


@pytest.mark.parametrize("ratio,b_occ,a,b_acts,phi,phi_b,chances,line", [
    (20.0, 650, 1.2, 800, 1.5, 1.3, 1, 0.004),
    (0.5, 30, 1.0, 30, 1.0, 1.0, 2, 0.01),
    (3.0, 0, 1.4, 0, 1.2, 1.0, 1, 0.003),
    (407256 / 20000, 368, 1.23, 420, 1.53, 1.28, 2, 0.0059),
])
def test_needed_occasions_is_the_first_count_that_passes(ratio, b_occ, a, b_acts, phi, phi_b, chances, line):
    n = TR.needed_occasions(line, ratio, b_occ, a, b_acts, phi, phi_b, chances, 3)
    assert n is not None and n >= 3
    assert _freq_p(n, ratio, b_occ, a, b_acts, phi, phi_b, chances) <= line
    if n > 3:
        assert _freq_p(n - 1, ratio, b_occ, a, b_acts, phi, phi_b, chances) > line


def test_needed_beyond_predicted_and_planner_invert_their_own_tests():
    k = TR.needed_beyond(0.006, 854, chances=2)
    assert 2 * TR.binom_above(k, 854, 0.05) <= 0.006 < 2 * TR.binom_above(k - 1, 854, 0.05)
    assert TR.needed_beyond(1e-300, 10) is None, "out of reach: even all 10 would not pass"
    k = TR.needed_predicted(0.02, 188)
    assert TR.binom_above(k, 188, 0.5) <= 0.02 < TR.binom_above(k - 1, 188, 0.5)
    # a touch row: against its own rate at matched no-touch moments (Fisher's test)
    k = TR.needed_predicted(0.02, 188, 6000, 27000)
    assert TR.two_share_above(k, 188, 6000, 27000) <= 0.02 < TR.two_share_above(k - 1, 188, 6000, 27000)
    assert TR.needed_predicted(0.5, 2) is None, "fewer judged occasions than the minimum of 3"
    k = TR.needed_planner_occasions(0.01, 40, 25000.0, 25000.0)
    assert TR.binom_above(k, k + 40, 0.5) <= 0.01 < TR.binom_above(k - 1, k - 1 + 40, 0.5)


def test_a_cell_says_what_passing_takes_and_how_far_away_it_is():
    p = max(TR.two_sample_above(680, 2.0, 327, 1.0), TR.two_sample_above_dispersed(816, 2.0, 1.2, 392, 1.0, 1.1))
    st = dict(reached=False, occasions=680, expected_by_chance=654.0, babble_occasions=327, acts=816,
              expected_acts=784.0, p_value=p, alpha=0.004, tests_together=100)
    cell = V.stage_text("repeated", st, dict(dispersion=1.2, babble_dispersion=1.1))
    assert cell["state"] == "not" and cell["marker"] == "Not yet"
    assert cell["text"].startswith("Not yet: 680 occasions against flailing's 654 in the same time; passing needs about ")
    n = int(cell["detail"].split("passing needs about ")[1].split(" ")[0].replace(",", ""))
    assert "(%s more)" % "{:,}".format(n - 680) in cell["detail"]
    # n is where this row's test reaches the line it would need with 100 tests in its family (q (k + 1) / m)
    need = 0.05 * (round(0.004 * 100 / 0.05) + 1) / 100
    assert TR.two_sample_above(n, 2.0, 327, 1.0) <= need and TR.two_sample_above_dispersed(n * 1.2, 2.0, 1.2, 392, 1.0, 1.1) <= need
    # a stage that waits keeps its number and says whether it would pass if judged today
    wait = V.stage_text("predicted_before_action", dict(reached=False, waits_for="repeated", predicted=693, judged=852,
                                                        p_value=1e-30, alpha=0.02), {})
    assert wait["state"] == "waits" and wait["marker"] == "Not judged yet" and wait["would_pass"] is True
    assert wait["text"] == "Not judged until Repeats passes. 693 of 852 occasions predicted; it would pass today."
    # failing its own test while the earlier stage has not passed either: not judged, and it says it would not pass
    early = V.stage_text("predicted_before_action", dict(reached=False, predicted=31, judged=188, p_value=0.3,
                                                         alpha=0.02), {}, dict(stages=dict(repeated=dict(reached=False))))
    assert early["state"] == "waits" and early["text"].startswith(
        "Not judged until Repeats passes. 31 of 188 occasions predicted; it would not pass today; passing needs about")
    kept = V.stage_text("retained", dict(reached=False, waits_for="repeated"), {})
    assert kept["text"] == "Not judged until Repeats passes. Nothing to measure until then."
    assert V.stage_text("repeated", dict(reached=None), {})["marker"] == "Not measurable"
    assert set(V.MARKERS.values()) == {"Passed", "Not yet", "Not judged yet", "Not measurable", "Not judged"}


def test_each_row_opens_with_one_sentence_on_how_close_it_is():
    def cells(*states):
        return [dict(key=k, state=s) for k, s in zip(V.MILESTONE_STAGES, states)]
    assert V.lead_sentence(cells("reached", "reached", "reached", "not")) == \
        "Closest to a full behavior: only Chosen is missing."
    assert V.lead_sentence(cells("reached", "not", "waits", "not")) == \
        "Passed Seen; a full behavior still needs Repeats, Predicted and Chosen."
    assert V.lead_sentence(cells("not", "not", "waits", "not")).startswith("No stage passed yet")
    assert V.lead_sentence(cells("unmeasured", "unmeasured", "unmeasured", "not")).startswith("Not measurable yet")
    assert V.lead_sentence(cells("reached", "reached", "reached", "reached")).startswith("A full behavior")


# ------------------------------------------------------------------ the dated copies
def _report(step, generated, reached=True, occ=100, exp=80.0, rules="tracker_rules_v5"):
    st = dict(reached=reached, first_step=step if reached else None, occasions=occ, expected_by_chance=exp)
    return dict(version="observer_analysis_v5", detector_version="observer_detectors_v10", rules_version=rules,
                generated=generated, last_step=step, baseline=dict(available=True, generated=1.0),
                capabilities={"H05": dict(id="H05", name="Individual finger isolation", group="Hand & finger",
                                          stages=dict(observed_once=st, repeated=dict(st)), counted=occ,
                                          subs={"L:thumb": dict(stages=dict(observed_once=dict(st, reached=False)))})})


def test_a_copy_every_4_hours_and_none_between(tmp_path):
    rep_path = tmp_path / "development.json"
    logs = []
    w = HI.SnapshotWriter(root=str(tmp_path / "hist"), log=logs.append)
    t0 = time.mktime(time.strptime("2026-09-30 08:00", "%Y-%m-%d %H:%M"))
    rep_path.write_text(json.dumps(_report(1000, t0)), encoding="utf-8")
    st = w.maybe(str(rep_path), "H99", now=t0, wait=True)
    assert st["ok"] and st["wrote"].endswith("development_20260930_0800_step1000.json")
    assert w.maybe(str(rep_path), "H99", now=t0 + 3600, wait=True) is st, "not due: nothing read, nothing written"
    rep_path.write_text(json.dumps(_report(2000, t0 + 4 * 3600)), encoding="utf-8")
    st = w.maybe(str(rep_path), "H99", now=t0 + 4 * 3600, wait=True)
    assert st["wrote"].endswith("development_20260930_1200_step2000.json")
    # a restarted daemon reads the folder: the newest copy is recent, so it waits
    w2 = HI.SnapshotWriter(root=str(tmp_path / "hist"), log=logs.append)
    st2 = w2.maybe(str(rep_path), "H99", now=t0 + 5 * 3600, wait=True)
    assert "wrote" not in st2 and w2.next_try["H99"] == pytest.approx(t0 + 8 * 3600)
    idx = HI.read_index(str(tmp_path / "hist" / "H99"))
    assert sorted(idx["entries"]) == ["development_20260930_0800_step1000.json", "development_20260930_1200_step2000.json"]
    assert idx["entries"]["development_20260930_0800_step1000.json"]["rows"]["H05"] == [100, 80.0, 3]


def test_copies_older_than_5_days_are_deleted_and_nothing_else(tmp_path):
    now = time.mktime(time.strptime("2026-09-30 12:00", "%Y-%m-%d %H:%M"))
    folder = tmp_path / "H99"
    folder.mkdir()
    old = HI.name_for(now - 6 * 86400, 10)
    young = HI.name_for(now - 4 * 86400, 20)
    rebuilt_old = HI.name_for(now - 5 * 86400 - 60, 30, rebuilt=True)
    for n in (old, young, rebuilt_old):
        HI.write_snapshot(json.dumps(_report(1, now)).encode(), str(folder), when=HI.parse_name(n)["time"],
                          rebuilt=n.endswith("_rebuilt.json"), rep=dict(_report(1, now), last_step=HI.parse_name(n)["step"]))
    (folder / "notes.json").write_text("{}")
    (folder / "development_old_by_hand.json").write_text("{}")
    gone = HI.prune(str(folder), now)
    assert sorted(gone) == sorted([old, rebuilt_old])
    assert sorted(os.listdir(folder)) == sorted([young, "notes.json", "development_old_by_hand.json", HI.INDEX])
    assert list(HI.read_index(str(folder))["entries"]) == [young]


def test_a_missing_history_disk_is_a_warning_and_a_retry_never_a_failure(tmp_path, monkeypatch):
    rep_path = tmp_path / "development.json"
    rep_path.write_text(json.dumps(_report(5, 100.0)), encoding="utf-8")
    monkeypatch.setattr(HI, "drive_present", lambda path: False)
    logs = []
    w = HI.SnapshotWriter(root=r"D:\UEMA_Research\external\D\UEMA_observer_history", log=logs.append)
    st = w.maybe(str(rep_path), "H99", now=1000.0, wait=True)
    assert st["ok"] is False and "missing" in st["note"] and len(logs) == 1 and "WARNING" in logs[0]
    assert w.next_try["H99"] == 1000.0 + HI.MISSING_RETRY_S
    w.maybe(str(rep_path), "H99", now=1000.0 + HI.MISSING_RETRY_S, wait=True)
    assert len(logs) == 1, "the warning is logged at most once an hour"
    # a failure inside the copy (a folder that cannot be made) is caught too
    monkeypatch.setattr(HI, "drive_present", lambda path: True)
    monkeypatch.setattr(HI.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError("device not ready")))
    st = w.maybe(str(rep_path), "H99", now=10 ** 6, wait=True)
    assert st["ok"] is False and "device not ready" in st["error"]
    # and the report file itself missing never raises either
    assert w.maybe(str(tmp_path / "nothing.json"), "H98", now=10 ** 6)["ok"] is False


# ------------------------------------------------------------------ the comparison
def test_compare_names_gains_losses_and_counts_and_flags_a_rule_change():
    old = _report(1000, 1000.0, reached=False, occ=50, exp=80.0)
    new = _report(2000, 5000.0, reached=True, occ=120, exp=96.0)
    head, changes = V.compare_view(new, None, old, "development_20260930_0800_step1000.json", [])
    assert changes["H05"] == ["Gained Seen and Repeats.",
                              "Occasions rose from 50 to 120; flailing's count for the same time rose from 80 to 96."]
    assert head["like_for_like"] is True and "1 row gained a stage, 0 rows lost one" in head["text"]
    # the other way round reads the same: always from the older report to the newer one
    assert V.compare_view(old, "development_20260930_0800_step1000.json", new, None, [])[1] == changes
    # a reset between the two dates (stage_history.jsonl) or different counting rules: not like for like
    reset = [dict(time=3000.0, step=1500, before="a", after="b")]
    # reports written before reports carried their counting rules are judged by the resets between them
    bare_new, bare_old = (dict(r, rules_version=None) for r in (new, old))
    head, _ = V.compare_view(bare_new, None, bare_old, None, reset)
    assert head["like_for_like"] is False and "Not like for like" in head["warning"] and "step 1,500" in head["warning"]
    head, _ = V.compare_view(_report(2000, 5000.0, rules="tracker_rules_v6"), None, old, None, [])
    assert head["like_for_like"] is False
    # both reports carry their counting rules (a copy rebuilt under today's rules): the rules decide, not the reset
    assert HI.compare_rules(dict(time=1.0, rules="x"), dict(time=9.0, rules="x"), reset)["like"] is True
    assert V.row_change(new["capabilities"]["H05"]["stages"], new["capabilities"]["H05"]["stages"]) is None


def test_rules_at_reads_the_stage_history_resets(tmp_path):
    lines = [dict(time=10.0, change="reset", step=5, reason="definitions, counting rules or babble changed: A -> B"),
             dict(time=11.0, cap="G01", change="lost", step=6),
             dict(time=20.0, change="reset", step=9, reason="definitions, counting rules or babble changed: B -> C")]
    (tmp_path / "stage_history.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    resets = HI.read_resets(str(tmp_path))
    assert [r["after"] for r in resets] == ["B", "C"]
    assert HI.rules_at(resets, 5.0) == "A" and HI.rules_at(resets, 15.0) == "B" and HI.rules_at(resets, 25.0) == "C"


def test_the_page_shows_a_dated_copy_and_compares_it_with_now(tmp_path, monkeypatch):
    saved = tmp_path / "Saved" / "mimo_bridge"
    (saved / "observer" / "H99").mkdir(parents=True)
    run = saved / "experiments" / "H99_seg1"
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(run))), encoding="utf-8")
    now = time.mktime(time.strptime("2026-09-30 16:00", "%Y-%m-%d %H:%M"))
    live = _report(2000, now, reached=True, occ=120, exp=96.0)
    live["exposure"] = {"0": dict(steps=9600)}
    (saved / "observer" / "H99" / "development.json").write_text(json.dumps(live), encoding="utf-8")
    hist = tmp_path / "hist"
    monkeypatch.setenv("UEMA_OBSERVER_HISTORY", str(hist))
    old = _report(1000, now - 8 * 3600, reached=False, occ=50, exp=80.0)
    path = HI.write_snapshot(json.dumps(old).encode(), str(hist / "H99"))
    name = os.path.basename(path)
    page = V.build(str(tmp_path), at=name, compare="now")
    assert page["shown"]["kind"] == "copy" and page["shown"]["file"] == name
    row = page["groups"][0]["rows"][0]
    assert row["changes"][0] == "Gained Seen and Repeats." and row["lead"].startswith("No stage passed yet")
    assert page["comparison"]["like_for_like"] is True and "step 1,000" in page["comparison"]["older"]
    hv = V.history_view(str(tmp_path))
    assert [e["file"] for e in hv["entries"]] == [name, "now"]
    assert hv["series"]["H05"] == [[0, 50, 80.0, 0], [1, 120, 96.0, 3]]
    assert V.build(str(tmp_path), at="../development.json")["available"] is False, "only the copies' own names"
    # every stage carries its plain question
    assert [s["question"] for s in page["stages"]][0] == "Does it happen more than random flailing?"
