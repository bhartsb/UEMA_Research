# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer statistics track of 2026-09-25 (docs/gain_of_function_v4/agents/observer_statistics_2026-09-25.md;
OBSERVER_DESIGN.md section 9): the defects fixed before the validation lock, each pinned on synthetic inputs
where the right answer is known, each failing on the code before its fix.
"""
import numpy as np
import pytest

from mimo_bridge.tools.observer import tracker as TR


def _det(step, body=0, cap="X", sub="", pred=None, well=None):
    ev = dict(prediction=None if pred is None else dict(kind="efference", verdict=pred), intention=None)
    if well is not None:
        ev["wellness"] = dict(total_change=well)
    return dict(cap=cap, sub=sub, key="k", body=body, step0=step, step1=step + 2, episode="e%d:%d" % (body, step),
                provenance="spontaneous", live=True, scripted=False, caregiver_touch=False, caregiver_present=False,
                posture="back", objects=[], measures={"m": 1.0}, evidence=ev)


# ------------------------------------------------------------------ the Benjamini-Hochberg line
def test_the_line_is_q_k_over_m_so_a_pass_shows_its_margin():
    """The same passes as the largest discovered p-value, but the weakest pass no longer sits exactly
    on its line, and a family whose p-values all underflow to 0 has the line q, not 0."""
    ps = [0.3, 0.045, 0.001, 0.02, 0.0124]          # 5% x k / 5: 0.01, 0.02, 0.03, 0.04, 0.05
    line = TR.fdr_line(ps)
    assert line == pytest.approx(0.05 * 3 / 5)
    assert sorted(p for p in ps if p <= line) == [0.001, 0.0124, 0.02]
    assert TR.fdr_line([0.0, 0.0, 0.0]) == pytest.approx(0.05)
    assert TR.fdr_line([0.2, 0.5]) == 0.0 and TR.fdr_line([]) == 0.0


def test_a_stage_is_dated_by_its_running_test_against_the_line_not_the_weakest_final_p():
    """Twenty rows with overwhelming evidence (every final p-value underflows to 0): dated against the
    old line (0) repeats waited until the running test itself underflowed; against q k / m it is
    dated by the third occasion's evidence, as a row judged alone is."""
    cat = dict(capabilities=[dict(id="C%d" % i, name="c%d" % i, group="g", quality=None) for i in range(20)])
    dets = [dict(_det(20 + 60 * k), cap="C%d" % i) for i in range(20) for k in range(200)]
    base = {"C%d" % i: dict(count=1, occasions=1, steps=10 ** 8) for i in range(20)}
    win = {str(w): 24000.0 for w in range(4)}
    ex = {"0": dict(steps=96000.0, by_author={"spontaneous": 96000.0}, windows=win, absent_steps=96000.0,
                    absent_by_author={"spontaneous": 96000.0}, absent_windows=win)}
    out = TR.track_all(dets, ex, base, cat)
    assert out["C3"]["stages"]["repeated"]["tests_together"] == 40      # 20 rows, each with the caregiver away
    rep = out["C3"]["stages"]["repeated"]
    assert rep["reached"] is True and rep["p_value"] == 0.0 and rep["alpha"] == pytest.approx(0.05)
    alone = TR.evaluate([d for d in dets if d["cap"] == "C3"], dict(ex["0"], bodies=1), base["C3"], None)
    assert rep["first_step"] == alone["stages"]["repeated"]["first_step"] < TR.RET_WINDOW


def test_a_stage_is_dated_where_its_running_test_last_went_under_the_line_not_at_an_early_dip():
    """Independent review of 2026-09-25, finding 2: with the higher v5 line a brief early dip of the running
    test dated a stage. Seen by size: the first two occasions start beyond babble's 95th percentile (the
    running test, doubled for the row's two chances, 0.005: under the line 0.05), eight do not (it rises to
    0.17), then eight more do: it goes under at step 320 and stays. Dated at 320, not 50."""
    from scipy import stats
    flags = [1, 1] + [0] * 8 + [1] * 8
    dets = []
    for k, f in enumerate(flags):
        d = _det(20 + 30 * k)
        d["measures"] = {"m": 3.0 if f else 1.0}
        dets.append(d)
    base = dict(count=5000, occasions=5000, steps=10000, measured_occasions=50,
                measures_occasion={"m": [0.1, 0.5, 2.0]})
    ex = dict(steps=1000, by_author={"spontaneous": 1000}, windows={})
    seen = TR.evaluate(dets, ex, base, ("m", "higher"))["stages"]["observed_once"]
    assert seen["reached"] is True and seen["by"] == "size"
    run = 2 * stats.binom.sf(np.cumsum(flags) - 1, np.arange(1, 19), 0.05)
    assert run[1] <= 0.05 and run[5:10].min() > 0.05 and (run[10:] <= 0.05).all(), "a dip, a rise, then under for good"
    assert seen["first_step"] == 320
    assert TR.last_stretch_start(run, 0.05, np.cumsum(flags) >= 1) == 10
    # Repeats (and Seen by frequency): five occasions in the first 100 steps pass at once against
    # babble's 1 in 1,000 steps; after two empty blocks the running test is above the line (p 0.88 at
    # step 9,020) until step 9,620, and under from there on
    W_ = TR.RET_WINDOW
    steps = [20 * (i + 1) for i in range(5)] + [3 * W_ + 20 + 60 * i for i in range(40)]
    win = {str(w): float(W_) for w in range(4)}
    ex2 = dict(steps=4.0 * W_, by_author={"spontaneous": 4.0 * W_}, windows=win)
    st = TR.evaluate([_det(s) for s in steps], ex2, dict(count=100, occasions=100, steps=100000), None)["stages"]
    early = TR.two_sample_above(3, 60.0, 100, 100000.0)
    assert early < 0.001, "the third occasion's running test was far under the line"
    assert st["repeated"]["reached"] is True and st["repeated"]["first_step"] == 9620
    assert st["observed_once"]["by"] == "frequency" and st["observed_once"]["first_step"] == 9620
    # a running test that ends above the line dates nothing
    assert TR.last_stretch_start(np.array([0.01, 0.2]), 0.05) is None
    assert TR.last_stretch_start(np.array([0.01, 0.02]), 0.05) == 0


def test_kept_counts_from_where_repeats_first_passed_so_a_faded_behaviour_stays_not_kept():
    """A behaviour repeats in the first block (30 occasions), makes none in the next two, and comes back
    (60 a block). Repeats is dated where its running test passed for good, in the fourth block (step
    10,060); Kept still counts from the block where that test first passed (step 140), so the two empty
    blocks mark it 'not kept'. Counted from the date shown, it would read 'kept' and hide the fade."""
    W_ = TR.RET_WINDOW
    steps = [20 + 60 * k for k in range(30)] + [w * W_ + 20 + 40 * k for w in (3, 4, 5) for k in range(60)]
    win = {str(w): 8.0 * W_ for w in range(6)}
    ex = dict(steps=48.0 * W_, by_author={"spontaneous": 48.0 * W_}, windows=win, bodies=8)
    out = TR.evaluate([_det(s) for s in steps], ex, dict(count=50, occasions=50, steps=10 ** 5), None)
    rep, kept = out["stages"]["repeated"], out["stages"]["retained"]
    assert rep["reached"] is True and rep["first_step"] == 10060 and rep["running_first_below_step"] == 140
    assert kept["reference_window"] == 0 and kept["reference_step"] == 140
    assert [(h["window"], h["count"], h["below"]) for h in kept["history"]] == [
        (1, 0, True), (2, 0, True), (3, 60, False), (4, 60, False), (5, 60, False)]
    assert kept["reached"] is False and out["regressed"] is False, "not kept; the newest block is fine"


def test_the_changed_rules_carry_new_versions():
    from mimo_bridge.tools.observer import daemon as DM
    # v6 (2026-10-08, observer_attribution_v1): an act of the hands alone counts by the writer class of every order in
    # force during it (test_observer_attribution.py)
    assert TR.RULES_VERSION == "tracker_rules_v6"
    assert DM.ANALYSIS_VERSION == "observer_analysis_v5"


# ------------------------------------------------------------------ 1 retention: windows judged together
W = TR.RET_WINDOW


def test_a_window_is_tested_against_the_reference_count_as_a_sample():
    """6 occasions in the first-repeat window, 0 in an equal later one: against 6 taken as the true
    rate, 0 is below half at p = e^-3 = 0.0498; with the reference's own noise it is not (p = (2/3)^6)."""
    assert TR.poisson_below(0, 0.5 * 6) == pytest.approx(np.exp(-3.0)) and TR.poisson_below(0, 3.0) < 0.05
    assert TR.rate_below_half(0, 1000.0, 6, 1000.0) == pytest.approx((2.0 / 3.0) ** 6)
    assert TR.rate_below_half(0, 1000.0, 6, 1000.0) > 0.05
    # a real fall from 40 to 0 stays far below the line; the same rate is never below
    assert TR.rate_below_half(0, 1000.0, 40, 1000.0) < 1e-6
    assert TR.rate_below_half(40, 1000.0, 40, 1000.0) > 0.5
    assert TR.rate_below_half(3, 0.0, 40, 1000.0) == 1.0


def _rows_with_windows(counts_by_row):
    """One capability per entry, each with the given occasions per window (occasions 60 steps apart
    on body 0, 4 windows of W steps with 8 bodies' exposure each, the caregiver away throughout)."""
    cat = dict(capabilities=[dict(id="C%d" % i, name="c%d" % i, group="g", quality=None)
                             for i in range(len(counts_by_row))])
    dets = []
    for i, counts in enumerate(counts_by_row):
        for w, n in enumerate(counts):
            dets += [_det(w * W + 20 + k * 60, cap="C%d" % i) for k in range(n)]
    base = {"C%d" % i: dict(count=1, occasions=1, steps=10 ** 8) for i in range(len(counts_by_row))}
    win = {str(w): 8.0 * W for w in range(4)}
    ex = {"0": dict(steps=32 * W, by_author={"spontaneous": 32 * W}, windows=win, absent_steps=32 * W,
                    absent_by_author={"spontaneous": 32 * W}, absent_windows=win)}
    return cat, dets, base, ex


def test_each_windows_retention_tests_are_judged_together_across_rows():
    """20 repeating rows; one fell from 40 occasions to none (a real drop), one from 40 to 9 (below
    half at p 0.016 alone), 18 kept 40. Alone at 5%, the second row reads 'not kept'; judged with its
    window's Benjamini-Hochberg line across the 20 rows (and their no-overseer columns) it does not,
    and the real drop still does."""
    rows = [[40, 40, 40, 0], [40, 40, 40, 9]] + [[40, 40, 40, 40]] * 18
    cat, dets, base, ex = _rows_with_windows(rows)
    alone = TR.evaluate([d for d in dets if d["cap"] == "C1"], dict(ex["0"], bodies=1), base["C1"], None)
    h_alone = alone["stages"]["retained"]["history"]
    assert h_alone[-1]["below"] is True and h_alone[-1]["p_value"] == pytest.approx(0.0159, abs=1e-4)
    assert alone["regressed"] is True, "alone it is marked regressed"
    out = TR.track_all(dets, ex, base, cat)
    real, marginal = out["C0"], out["C1"]
    assert real["stages"]["retained"]["reference_window"] == 0
    assert real["regressed"] is True and real["stages"]["retained"]["reached"] is False
    assert marginal["regressed"] is False and marginal["stages"]["retained"]["reached"] is True
    last = marginal["stages"]["retained"]["history"][-1]
    assert last["rate_low"] is True and last["below"] is False
    assert last["tests_together"] == 40 and last["alpha"] < last["p_value"] < 0.05
    assert out["C5"]["stages"]["retained"]["reached"] is True and not out["C5"]["regressed"]
    # the no-overseer column is judged in the same families
    assert out["C0"]["no_overseer"]["retained"]["reached"] is False
    assert out["C1"]["no_overseer"]["retained"]["reached"] is True
    # the stage names its lowest window's p-value against that window's line
    st = marginal["stages"]["retained"]
    assert st["worst_window_p"] == pytest.approx(last["p_value"]) and st["worst_window_alpha"] == last["alpha"]


# ------------------------------------------------------------------ 7 the p-value against its line in the cell's tooltip
def test_a_stage_cells_tooltip_gives_its_p_value_against_its_line():
    from mimo_bridge.tools.observer import view as V
    st = dict(reached=True, first_step=388777, p_value=0.00292, alpha=0.00365, tests_together=79, occasions=160,
              expected_by_chance=100.0)
    cell = V.stage_text("repeated", st, {})
    assert cell["state"] == "reached" and cell["text"].startswith("Passed at step 388,777")
    # chances as pictured counts over one shared number (independent review of 2026-09-25, finding 4)
    assert ": 29 in 10,000." in cell["tip"] and ": 36 in 10,000." in cell["tip"] and "79 tests" in cell["tip"]
    assert "passes at or under its line" in cell["tip"]
    assert V._chances(0.00364, 0.00365) == ("36.4 in 10,000", "36.5 in 10,000"), "a decimal where both round alike"
    # Seen's p-value is doubled only where both of its chances were tested (73 of 156 tested cells on the
    # lineage had the frequency test alone)
    one = V.stage_tip("observed_once", dict(p_value=0.001, alpha=0.01, rate_p=0.001, share_p=None, tests_together=9))
    two = V.stage_tip("observed_once", dict(p_value=0.002, alpha=0.01, rate_p=0.001, share_p=0.3, tests_together=9))
    assert "doubled" not in one and "doubled because the row had two chances" in two
    # tiny p-values are said per million; a row judged alone says so
    tiny = V.stage_text("predicted_before_action", dict(reached=True, first_step=5, p_value=2.1e-5, alpha=0.05,
                                                        predicted=9, judged=10), {})
    assert "21 in a million" in tiny["tip"] and "judged alone" in tiny["tip"]
    assert "under 1 in a million" in V.stage_tip("repeated", dict(p_value=0.0, alpha=0.05, tests_together=3))
    # a stage that waits keeps its own test in the tooltip and says it is not reached
    wait = V.stage_text("predicted_before_action", dict(reached=False, waits_for="repeated", p_value=0.001,
                                                        alpha=0.02, tests_together=40, predicted=47, judged=60), {})
    assert wait["state"] == "waits" and "not reached until Repeats is" in wait["tip"]
    # no test, no tooltip: unmeasured, a planner stage never tested, Varied starts
    assert "tip" not in V.stage_text("repeated", dict(reached=None), {})
    assert "tip" not in V.stage_text("selected_intentionally", dict(reached=False, decisive=0), {})
    assert "tip" not in V.stage_text("generalized", dict(reached=True, first_step=5, bodies=4, contexts=2), {})
    # Kept names its lowest block's p-value against that block's line
    kept = dict(reached=False, windows_after=7, worst_window=131, worst_window_p=1.3e-9, worst_window_alpha=0.0063,
                history=[dict(window=130, below=False, tests_together=8), dict(window=131, below=True, tests_together=8),
                         dict(window=136, below=False, tests_together=8)])
    cell = V.stage_text("retained", kept, {})
    assert cell["text"].startswith("Not kept") and "under 1 in a million" in cell["tip"] and "6 in 1,000" in cell["tip"]
    assert "8 rows" in cell["tip"] and "a count this low or lower" in cell["tip"]


def test_a_part_shows_its_own_regression_and_the_watch_lists_it(tmp_path):
    """A part's Kept cell reads its own newest block, not the whole row's flag; the regression watch
    lists a regressed part and counts parts among the checked."""
    import json
    from mimo_bridge.tools.observer import view as V
    own = dict(reached=False, windows_after=3, history=[dict(window=1, below=False), dict(window=2, below=True)])
    fine = dict(reached=True, windows_after=3, history=[dict(window=1, below=False), dict(window=2, below=False)])
    assert V.stage_text("retained", own, dict(regressed=False))["text"].startswith("Regressed")
    assert V.stage_text("retained", fine, dict(regressed=True))["text"].startswith("Passed"), \
        "the row's flag does not make a part that kept its rate read as regressed"
    saved = tmp_path / "Saved" / "mimo_bridge"
    (saved / "observer" / "H99").mkdir(parents=True)
    run = saved / "experiments" / "H99_seg1"
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(run))), encoding="utf-8")
    caps = {"G02": dict(id="G02", name="Fixation on object", group="Gaze", counted=90,
                        stages=dict(retained=fine), regressed=False,
                        subs={"arm_L_upper": dict(counted=80, regressed=True, stages=dict(retained=own))})}
    rep = dict(capabilities=caps, exposure={"0": dict(steps=9600)}, baseline=dict(available=True, steps=20000),
               counts=dict(live_detections=90))
    (saved / "observer" / "H99" / "development.json").write_text(json.dumps(rep), encoding="utf-8")
    page = V.build(str(tmp_path))
    assert [r.get("sub") for r in page["regressions"]] == ["arm_L_upper"]
    assert "left upper arm" in page["regressions"][0]["text"]
    caps["G02"]["subs"]["arm_L_upper"].update(regressed=False, stages=dict(retained=fine))
    (saved / "observer" / "H99" / "development.json").write_text(json.dumps(rep), encoding="utf-8")
    page = V.build(str(tmp_path))
    assert page["regressions"] == [] and "(2 checked)" in page["regression_note"]


# ------------------------------------------------------------------ 3 and 5 the fast lanes
class _FastTrack:
    """The fields of features.Track the fast lane reads: 5 frames a record, a still head, the body's
    weight on the floor."""

    def __init__(self, first_step, n, weight=30.0):
        self.n = n
        self.step = np.arange(first_step, first_step + n)
        self.frames = np.full(n, 5)
        self.header = dict(weight_n=weight)
        self.fast = np.full((n, 5, 8), np.nan)
        self.fast[:, 0, :5] = 9.81            # the head's acceleration: gravity only
        self.fast[:, 1, :5] = 0.2             # its turning speed
        self.fast[:, 2, :5] = weight          # the force from outside: the body's weight
        self.fast[:, 3, :5] = 5.0             # the largest contact force
        self.fast[:, 4, :5] = 0.0


def _digest(tmp_path, name="H99_x_seg1"):
    from mimo_bridge.tools.observer import daemon as DM
    run = tmp_path / name
    (run / "observer" / "world" / "b0").mkdir(parents=True)
    return DM.RunDigest(str(run))


def test_a_fast_spike_is_counted_once_in_the_bin_of_its_peak(tmp_path):
    from mimo_bridge.tools.observer import daemon as DM
    from mimo_bridge.tools.observer import fastlane as FL
    assert DM.FAST_BIN == DM.BIN == 100, "the fast lanes use the exposure's bins"
    tr = _FastTrack(0, 300)
    tr.fast[99, 0, 2] = 9.81 + 20.0          # a 20 m/s2 jolt at record 99 (step 99, bin 0)
    tr.fast[150, 0, 1] = 9.81 + 8.0          # an 8 m/s2 jolt at record 150 (bin 1)
    dg = _digest(tmp_path)
    dg._fast(0, 7, tr, np.arange(100, 200))  # this window finalizes steps 100-199 only
    fb = dg.fast["0"]["7"]
    assert set(fb) == {"1"} and fb["1"]["head_jolt"]["samples"] == 500
    assert fb["1"]["head_jolt"]["spikes"] == 1 and fb["1"]["head_jolt"]["largest"] == pytest.approx(8.0)
    dg._fast(0, 7, tr, np.arange(0, 100))    # the window that finalizes steps 0-99 counts the first
    assert fb["0"]["head_jolt"]["spikes"] == 1 and fb["0"]["head_jolt"]["largest"] == pytest.approx(20.0)
    assert fb["1"]["head_jolt"]["spikes"] == 1, "counted once, in the window that finalizes its peak"
    assert dg.state["fast_bin"] == 100
    # the helper alone: a spike whose peak record is not kept is another window's
    x = np.ones(50)
    x[20] = 30.0
    rec = np.arange(50)
    assert FL.binned(x, 0.05, rec, rec >= 25, rec // 10)[2]["spikes"] == 0
    assert FL.binned(x, 0.05, rec, rec < 25, rec // 10)[2]["spikes"] == 1


def test_the_learners_fast_lane_reads_the_lane_one_record_later(tmp_path):
    """Lane step t describes world record t - 1: a spike at lane step 1000 belongs to world record
    999 (bin 9), not to record 1000 (bin 10), as it did with lane step = world step."""
    dg = _digest(tmp_path)

    class _Lane:
        senses = ["tactile"]
        rows = {s: dict(next_pe_body=np.array([[1.0 + 0.01 * (s % 7)]])) for s in range(900, 1101)}
    _Lane.rows[1000] = dict(next_pe_body=np.array([[40.0]]))
    dg.lane = _Lane()
    tr = _FastTrack(900, 200)
    dg._fast(0, 7, tr, np.arange(0, 200))
    fb = dg.fast["0"]["7"]
    assert fb["9"]["learner_tactile"]["spikes"] == 1 and fb["9"]["learner_tactile"]["short_peak"] == pytest.approx(40.0)
    assert fb["10"]["learner_tactile"]["spikes"] == 0


def test_a_crash_resumes_steps_count_once_in_the_fast_lanes(tmp_path, monkeypatch):
    """Run A recorded steps 0-599; the crash resume B restarted from step 400 and recorded 400-799.
    Exposure counted A's steps 400-599 once (the lineage cut); the fast lanes summed both copies."""
    import json
    from mimo_bridge.tools.observer import catalog as CAT
    from mimo_bridge.tools.observer import daemon as DM
    from mimo_bridge.tools.observer import store as STO
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "saved"))
    runs = []
    for name, first, last, ms in (("H99_x_seg1", 0, 599, 1000), ("H99_x_seg1_resume_1", 400, 799, 2000)):
        run = tmp_path / name
        (run / "observer" / "world" / "b0").mkdir(parents=True)
        (run / "observer" / "world" / "b0" / ("header_7_%d.json" % ms)).write_text(json.dumps(dict(scene="crib")),
                                                                                     encoding="utf-8")
        a = run / "observer" / "analysis"
        a.mkdir(parents=True)
        bins = {str(b): dict(steps=100, by={"spontaneous": 100}, abs=0, abs_by={}) for b in range(first // 100, last // 100 + 1)}
        fast = {str(b): dict(head_jolt=dict(samples=500, spikes=1, largest=3.0, area_sum=0.1, recovery_sum_s=0.02))
                for b in range(first // 100, last // 100 + 1)}
        st = dict(version=DM.ANALYSIS_VERSION, detector_version=DM.DETECTOR_VERSION, first_step=first, start_ms=ms,
                  cuts={}, fast_bin=100, bodies={})
        for fn, v in (("state.json", st), ("exposure.json", {"0": {"7": bins}}), ("fastlane.json", {"0": {"7": fast}})):
            (a / fn).write_text(json.dumps(v), encoding="utf-8")
        runs.append(str(run))
    store = STO.LineageStore(None, DM.quality_map(CAT.load()))
    rep = DM.build_report(runs, str(tmp_path / "out" / "development.json"), None, render_replays=False, store=store)
    assert rep["recording"]["lineage_cuts"] == {"H99_x_seg1": 400}
    assert rep["exposure"]["0"]["steps"] == 800, "exposure: steps 0-799 once"
    jolt = rep["fast_lane_total"]["head_jolt"]
    assert jolt["samples"] == 8 * 500 and jolt["spikes"] == 8, "the fast lane: the same 8 bins of 100 steps"


def test_a_cut_inside_a_bin_counts_only_the_records_before_it(tmp_path, monkeypatch):
    """Independent review of 2026-09-25, finding 6: crash resumes restart at multiples of 25 steps, so a
    lineage cut can fall inside a 100-step bin. Run A recorded 0-599, its crash resume B restarted at 425:
    A's bin 400-499 holds 25 records before the cut and 75 after it, which B recorded again. The whole
    bin was counted (875 body-steps for 800); now its first 25 records are (A's first and last step in
    the bin, 400 and 499, are kept since analysis v5)."""
    import json
    from mimo_bridge.tools.observer import catalog as CAT
    from mimo_bridge.tools.observer import daemon as DM
    from mimo_bridge.tools.observer import store as STO
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "saved"))
    runs = []
    for name, first, last, ms in (("H99_x_seg1", 0, 599, 1000), ("H99_x_seg1_resume_1", 425, 799, 2000)):
        run = tmp_path / name
        (run / "observer" / "world" / "b0").mkdir(parents=True)
        (run / "observer" / "world" / "b0" / ("header_7_%d.json" % ms)).write_text(json.dumps(dict(scene="crib")),
                                                                                     encoding="utf-8")
        a = run / "observer" / "analysis"
        a.mkdir(parents=True)
        bins, fast = {}, {}
        for b in range(first // 100, last // 100 + 1):
            lo, hi = max(first, 100 * b), min(last, 100 * b + 99)
            n = hi - lo + 1
            bins[str(b)] = dict(steps=n, by={"spontaneous": n}, abs=0, abs_by={}, lo=lo, hi=hi)
            fast[str(b)] = dict(head_jolt=dict(samples=5 * n, spikes=4, largest=3.0, area_sum=0.4, recovery_sum_s=0.08))
        st = dict(version=DM.ANALYSIS_VERSION, detector_version=DM.DETECTOR_VERSION, first_step=first, start_ms=ms,
                  cuts={}, fast_bin=100, bodies={})
        for fn, v in (("state.json", st), ("exposure.json", {"0": {"7": bins}}), ("fastlane.json", {"0": {"7": fast}})):
            (a / fn).write_text(json.dumps(v), encoding="utf-8")
        runs.append(str(run))
    store = STO.LineageStore(None, DM.quality_map(CAT.load()))
    rep = DM.build_report(runs, str(tmp_path / "out" / "development.json"), None, render_replays=False, store=store)
    assert rep["recording"]["lineage_cuts"] == {"H99_x_seg1": 425}
    assert rep["exposure"]["0"]["steps"] == pytest.approx(800), "steps 0-799 once, not 875"
    jolt = rep["fast_lane_total"]["head_jolt"]
    assert jolt["samples"] == 5 * 800, "the fast lane's frames: A's 0-424 and B's 425-799"
    assert jolt["spikes"] == 4 * 4 + 1 + 4 * 4, "A's straddling bin keeps a quarter of its 4 spikes"
    assert DM._before_cut(400, 100, 425, 400, 499) == 0.25 and DM._before_cut(400, 100, 425) == 0.25
    assert DM._before_cut(400, 100, 500) == 1.0 and DM._before_cut(400, 100, 400) == 0.0
    assert DM._before_cut(400, 100, 440, 420, 459) == 0.5, "records 420-459: 20 of 40 before the cut"
    # a run that ended mid-bin before the next run's first step (a planned continuation) keeps its records
    assert DM._before_cut(397300, 100, 397350, 397300, 397349) == 1.0
    assert DM._before_cut(397300, 100, 397350, records=50) == 1.0, "a bin without lo and hi: its 50 records"


# ------------------------------------------------------------------ 2 novelty: the within-unit order control
def _brute_order_chance(tokens, run):
    """The share of all orders of `tokens` whose runs (novelty._runs) include `run`."""
    import itertools
    from mimo_bridge.tools.observer import novelty as NV
    hits = total = 0
    for perm in itertools.permutations(tokens):
        total += 1
        hits += int(tuple(run) in set(NV._runs(list(perm))))
    return hits / total


@pytest.mark.parametrize("tokens,run", [
    (["a", "b"], ("a", "b")),
    (["a", "a", "b"], ("a", "b")),
    (["a", "a", "b", "x", "y"], ("a", "b")),
    (["a", "b", "b", "a", "x", "x"], ("b", "a")),
    (["a", "b", "c"], ("a", "b", "c")),
    (["a", "b", "b", "c"], ("a", "b", "c")),
    (["a", "a", "b", "b", "c", "x", "y"], ("a", "b", "c")),
    (["a", "b", "c", "c", "b", "a", "x"], ("c", "b", "a")),
])
def test_the_order_chance_is_exact(tokens, run):
    from collections import Counter
    from mimo_bridge.tools.observer import novelty as NV
    c = Counter(tokens)
    assert NV.order_chance(len(tokens), tuple(c[t] for t in run)) == pytest.approx(_brute_order_chance(tokens, run),
                                                                                   abs=1e-12)


def _unit(i, body, tokens, caps=()):
    return dict(id="u%d" % i, body=body, step0=i * 100, step1=i * 100 + 20, tokens=list(tokens), caps=list(caps),
                live=True, caregiver_touch=False, authors={2: 20})


def test_a_pair_that_busy_units_make_by_chance_is_not_a_candidate():
    """60 busy units of six distinct touches each, in random orders: 'a then b' comes in about 10 of
    them only because every unit holds both. It recurs (5+ units, 2+ bodies) but no more than the
    same units' tokens in any order would show it. 'p then q', written in that order in 12 units of
    two tokens, is kept: shuffled, each shows it half the time."""
    from mimo_bridge.tools.observer import novelty as NV
    rng = np.random.default_rng(5)
    kinds = ["touch:hand_R_palm~%s" % k for k in ("box_light", "box_medium", "box_heavy", "target", "mobile", "wall")]
    a, b = kinds[0], kinds[1]
    units = [_unit(i, i % 4, list(rng.permutation(kinds))) for i in range(60)]
    p_, q_ = "object_moves:box_light", "lift:box_light"
    units += [_unit(100 + i, i % 3, [p_, q_]) for i in range(12)]
    tests = {}
    kept = NV.mine(units, tests=tests)
    ids = {c["id"]: c for c in kept}
    chance_ab = NV.candidate_id((a, b))
    seen_ab = sum(1 for u in units[:60] if (a, b) in set(NV._runs(u["tokens"])))
    assert seen_ab >= NV.MIN_EPISODES, "the chance pair recurs"
    assert chance_ab not in ids, "a pair busy units make by chance is no candidate"
    pq = ids[NV.candidate_id((p_, q_))]
    assert pq["episodes"] == 12 and pq["order_expected"] == pytest.approx(6.0) and pq["order_p"] == pytest.approx(0.5 ** 12)
    assert pq["chance_measured"] is False and pq["combined_p"] == pq["order_p"]
    assert tests["tested"] >= 2 and tests["order_measured"] == tests["tested"] and tests["kept"] == len(kept)
    # with a babble baseline both controls must pass: babble making p-then-q as often drops it
    loud = NV.mine(units, baseline=dict(runs={pq["id"]: 120}, steps=10000), exposure_steps=1000)
    assert pq["id"] not in {c["id"] for c in loud}
    quiet = NV.mine(units, baseline=dict(runs={}, steps=10000), exposure_steps=1000)
    got = {c["id"]: c for c in quiet}[pq["id"]]
    assert got["combined_p"] == pytest.approx(max(got["p_value"], got["order_p"]))


def test_a_novelty_candidates_sentence_names_its_order_control(tmp_path):
    import json
    from mimo_bridge.tools.observer import view as V
    saved = tmp_path / "Saved" / "mimo_bridge"
    (saved / "observer" / "H99").mkdir(parents=True)
    run = saved / "experiments" / "H99_seg1"
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(run))), encoding="utf-8")
    cand = dict(label="novel candidate #ab12cd34", tokens=["object_moves:box_light", "lift:box_light"], episodes=12,
                bodies=3, later_share=0.5, world_effect=True, chance_measured=True, expected_by_chance=0.4,
                order_expected=6.0, examples=[])
    rep = dict(capabilities={}, exposure={"0": dict(steps=9600)}, baseline=dict(available=True, steps=20000),
               counts=dict(live_detections=0), novelty=[cand])
    (saved / "observer" / "H99" / "development.json").write_text(json.dumps(rep), encoding="utf-8")
    text = V.build(str(tmp_path))["novelty"][0]["text"]
    assert "random flailing: 0.4 in the same time" in text and "in a random order, 6 would show it" in text


# ------------------------------------------------------------------ found: a window's look-back
def test_a_window_reads_ten_chunks_before_its_first_unfinalized_one(tmp_path, monkeypatch):
    """Turn acts depend on where the read record starts (primitives._turns scans from its first
    record), so the live digester, a chunk or two per window, counted 3.4% fewer of them than one pass
    over the same records when a window read one chunk before its new ones; with ten, 0.28%."""
    from mimo_bridge.tools.observer import daemon as DM
    from mimo_bridge.tools.observer.stream import ChunkRef, Segment
    assert DM.LOOKBACK_CHUNKS == 10
    seg = Segment(pid=7, start_ms=1, chunks=[ChunkRef("c%02d.npz" % k, 100 * k, 7, k) for k in range(15)])
    dg = _digest(tmp_path)
    monkeypatch.setattr(dg.files, "world_segments", lambda body: [seg])
    monkeypatch.setattr(DM, "read_chunk", lambda path: dict(header_name="h", step=np.arange(
        100 * int(path[1:3]), 100 * int(path[1:3]) + 100)))
    seen = []

    def process(body, pid, group, closed, cut):
        seen.append([c.name for c, _ in group])
        for c, d in group:
            dg.state["bodies"][str(body)]["done"][c.name] = dict(pid=pid)
        return dict(detections=0, episodes=0, chunks=len(group))
    monkeypatch.setattr(dg, "_process", process)
    dg.state["bodies"]["0"] = dict(done={"c%02d.npz" % k: {} for k in range(14)}, final={"7": 1399})
    dg._digest_body(0, final=True)
    assert seen == [["c%02d.npz" % k for k in range(4, 15)]], "ten chunks of look-back and the new one"


# ------------------------------------------------------------------ 4 a hand or arm act's order lags its motion
class _CreditTrack:
    """Order 7 (the planner, tied) at records 0-9, order 8 (the planner, decisive) from record 10; the
    part moves 0.2 cm a record, and 3 cm at record 10, the new order's FIRST record."""

    def __init__(self, n=30, nbody=4):
        self.n = n
        self.step = np.arange(1000, 1000 + n)
        self.order_seq = np.r_[np.full(10, 7), np.full(n - 10, 8)]
        self.author = np.full(n, 1, dtype=np.int64)
        self.xpos = np.zeros((n, nbody, 3))
        for i in range(1, n):
            self.xpos[i] = self.xpos[i - 1] + 0.002
        self.xpos[10:] += 0.03
        self.eye_fwd = np.tile(np.array([1.0, 0.0, 0.0]), (n, 2, 1))
        self.continuous = np.ones(n, dtype=bool)
        self.continuous[0] = False


CREDIT_HEADER = dict(patch_bodies=[], entities=[], handle_names=["index_R", "thigh_L", "calf_L", "foot_L"],
                     handle_bodies=[0, 1, 2, 3])


def _latch(step, seq, outcome):
    return dict(type="latch", step=step, order_seq=seq, reached_body=True,
                plan=dict(pred_wellness_gaps=dict(outcome=[outcome], gap_to_best_alt=[0.2], tie_band=0.0734)))


def test_a_hand_act_is_credited_to_no_single_order_and_a_leg_act_to_its_peak():
    """Until observer_attribution_v1 (2026-10-08) a hand act was credited to the order one record before its peak
    motion (the lag rule, CREDIT_LAG_RECORDS = 1): this act read order 7. Now an act of the hands alone names no single
    order: it lists orders 7 and 8 with the movement made during each (test_observer_attribution.py), and its intention
    needs every one of them; a leg act still takes the order at its peak."""
    from mimo_bridge.tools.observer import evidence as EV
    assert not hasattr(EV, "CREDIT_LAG_RECORDS"), "the lag rule is gone"
    tr = _CreditTrack()
    j = EV.Joiner(None, [_latch(1000, 7, "tied"), _latch(1010, 8, "best")], ["index_R"], CREDIT_HEADER)
    finger = dict(cap="H05", sub="R:index", step0=1008, step1=1014, i0=8, i1=14, parts=["finger_R_index"])
    cr = j.credit(finger, tr)
    assert cr["rule"] == "command_intervals" and "order_seq" not in cr and cr["commands"] == [7, 8]
    att = j.command_intervals(finger, tr)
    assert [(e["command"], e["records"]) for e in att["intervals"]] == [(7, [1009, 1009]), (8, [1010, 1014])]
    assert j.intention_intervals(finger, 0, att)["verdict"] is False, "order 7, in force during the act, tied"
    # the legs follow their order at once: the same motion of a leg act is order 8's
    leg = dict(cap="P09", step0=1008, step1=1014, i0=8, i1=14, parts=["leg_L"])
    cl = j.credit(leg, tr)
    assert cl["peak_step"] == cl["credit_step"] == 1010 and cl["order_seq"] == 8 and cl["lag_records"] == 0
    assert j.intention(leg, 0, cl)["verdict"] is True
    # an act of the hands alone, by its parts' handles (formerly `lags`, the same test under its new name)
    assert EV.Joiner.is_hand_act(dict(cap="H05", parts=["finger_R_index"])) and EV.Joiner.lags(dict(cap="H07", parts=["hand_R"]))
    assert not EV.Joiner.is_hand_act(dict(cap="B06", parts=["arm_R"]))
    assert not EV.Joiner.is_hand_act(dict(cap="R01", parts=["hand_R", "arm_R"]))


# ------------------------------------------------------------------ 6 the wellness change against its reference
class _WellLane:
    """Wellness rows in world numbering (lane lag 0): one channel rising 0.001 a step for body 0, and
    0.05 more across the act's own records 1060-1070."""

    def __init__(self):
        self.rows = {}
        for s in range(990, 1130):
            v = 0.001 * (s - 990) + (0.05 if 1060 < s <= 1070 else 0.0) + (0.05 if s > 1070 else 0.0)
            self.rows[s] = dict(wellness={"coherence": np.array([v, 0.0])})


class _WellTrack:
    def __init__(self, n=120, first=1000):
        self.n = n
        self.step = np.arange(first, first + n)
        self.live = np.ones(n, dtype=bool)
        self.continuous = np.ones(n, dtype=bool)
        self.continuous[0] = False


def test_the_wellness_change_is_measured_against_equal_stretches_just_before_the_act(monkeypatch):
    """Analysis v5 (independent review of 2026-09-25, finding 1): an act's reference stretches have its
    length and end 1 to 3 s before it starts, one a second, so they depend on the act alone and not on
    the digestion window (until v5 they were the moments of the steps a window finalized, given to the
    nearest act of that window: 35 against 65 in 100 for one row's touch control, batch against live)."""
    from mimo_bridge.tools.observer import evidence as EV
    j = EV.Joiner(None, [], [], dict(patch_bodies=[], entities=[]))
    j.lane = _WellLane()
    tr = _WellTrack()
    act = dict(cap="H05", step0=1060, step1=1070)
    earlier = dict(cap="H05", step0=1022, step1=1028)   # an act before it, found in the window's look-back
    later = dict(cap="H05", step0=1080, step1=1090)     # an act after it, which a live window may not have yet
    act["evidence"] = dict(wellness=j.wellness(act, 0))
    assert act["evidence"]["wellness"]["total_change"] == pytest.approx(0.06)
    j.wellness_references([act], 0, tr, all_dets=[earlier, act, later])
    wl = act["evidence"]["wellness"]
    # stretches of 10 steps from 1040, 1020 and 1000 (ending 10, 30 and 50 steps before the act), each
    # 0.001 x 10 = 0.01; the one from 1020 holds the earlier act: in ALL, not WITHOUT
    assert wl["all_n"] == 3 and wl["all_sum"] == pytest.approx(0.03) and wl["all_sq"] == pytest.approx(3e-4)
    assert wl["reference_n"] == 2 and wl["reference_sum"] == pytest.approx(0.02) and wl["reference_sq"] == pytest.approx(2e-4)
    assert wl["reference_reach_steps"] == EV.MATCH_REACH_STEPS == 60
    # a window that has not found the later act gives the same references
    act2 = dict(act, evidence=dict(wellness=j.wellness(act, 0)))
    j.wellness_references([act2], 0, tr, all_dets=[earlier, act2])
    assert {k: act2["evidence"]["wellness"][k] for k in ("reference_n", "all_n")} == dict(reference_n=2, all_n=3)
    # an act longer than REFERENCE_MAX_STEPS gets none (its stretches would start before the look-back)
    monkeypatch.setattr(EV, "REFERENCE_MAX_STEPS", 5)
    act3 = dict(act, evidence=dict(wellness=j.wellness(act, 0)))
    j.wellness_references([act3], 0, tr, all_dets=[act3])
    assert act3["evidence"]["wellness"]["all_n"] == 0 and act3["evidence"]["wellness"]["reference_n"] == 0


def _well_dets(act_change, without_mean, all_mean):
    """12 occasions whose first act changed wellness by act_change (+-0.002), each given 4 stretches
    without the act (mean without_mean) and 6 stretches at any moment (mean all_mean), with a burst's
    second act (0.5) that is no occasion's first."""
    dets = []
    for i in range(12):
        d = _det(1000 + 100 * i, well=act_change + 0.002 * (i % 3))
        d["evidence"]["wellness"].update(reference_n=4, reference_sum=4 * without_mean,
                                         reference_sq=4 * without_mean ** 2 + 0.00001 * (i % 2),
                                         all_n=6, all_sum=6 * all_mean, all_sq=6 * all_mean ** 2 + 0.00001 * (i % 2))
        dets.append(d)
        dets.append(_det(1000 + 100 * i + 3, well=0.5))
    return dets


def test_the_tracker_puts_the_acts_wellness_change_beside_both_references():
    """The act's change against the stretches just before it, whatever happened in them and without the
    act; each occasion is one pair (the act minus the mean of its own stretches), so stretches that overlap
    because the act is long are not counted as separate samples (independent review of 2026-09-25,
    finding 4). The page shows one plain line and puts the numbers in its tooltip; where the act changes
    wellness less than the stretches without it while those rose more than the stretches in general, the
    tooltip says that this is not by itself a cost of the act."""
    from scipy import stats
    from mimo_bridge.tools.observer import store as ST
    from mimo_bridge.tools.observer import view as V
    ex = dict(steps=10000, by_author={"spontaneous": 10000}, windows={}, bodies=8)
    w = TR.evaluate(_well_dets(0.002, 0.012, 0.001), ex, None, None)["wellness"]
    assert w["acts_n"] == 12 and w["acts_mean"] == pytest.approx(0.004)
    assert (w["reference_n"], w["all_n"], w["reference_stretches"], w["all_stretches"]) == (12, 12, 48, 72)
    assert w["reference_mean"] == pytest.approx(0.012) and w["all_mean"] == pytest.approx(0.001)
    assert w["difference"] == pytest.approx(-0.008) and w["difference_all"] == pytest.approx(0.003)
    # the interval from the spread of the 12 occasions' differences (0.002, 0.004, 0.006, four times each)
    half = stats.t.ppf(0.975, 11) * np.std([0.002, 0.004, 0.006] * 4, ddof=1) / np.sqrt(12)
    assert w["difference_ci95"] == pytest.approx(half) and w["difference_all_ci95"] == pytest.approx(half)
    assert w["in_act_share"] == pytest.approx(1.0 / 3.0)
    view = V.wellness_view(w)
    assert view["text"].startswith("Wellness (the sum of the learner's grounded wellness channels, each 0 to 1) "
                                   "went up more (or down less) during this act than in the same length of time just "
                                   "before it.")
    assert "Against only the stretches in which the act did not happen, wellness went up less" in view["text"]
    assert sum(ch.isdigit() for ch in view["text"].replace("0 to 1", "")) == 0, "no numbers in the row's line"
    tip = view["tip"]
    assert "The first act of each of 12 occasions changed wellness by +0.004 on average" in tip
    assert "72 stretches; the same behaviour went on in 33 in 100 of them" in tip
    # a row that runs almost without pause says so in its line: the time before one act is mostly another
    assert "already going on in 93 of every 100 of those stretches" in V.wellness_view(dict(w, in_act_share=0.93))["text"]
    assert "the act minus its own stretches is +0.003" in tip and "the act minus those is -0.008" in tip
    assert "is not by itself a cost of the act" in tip
    # an act above the stretches without it needs no such clause; one reference alone is never shown
    up = TR.evaluate(_well_dets(0.05, 0.01, 0.001), ex, None, None)["wellness"]
    assert up["difference"] > 0 and "cost" not in V.wellness_view(up)["tip"]
    # nor a difference inside its interval (the eye movement row on the lineage: -0.00077, 95% interval
    # -0.0019 to +0.0004, with the stretches without the act a hair above the stretches in general)
    tiny = dict(w, difference=-0.00077, difference_ci95=0.00115, reference_mean=-0.00138, all_mean=-0.00141,
                difference_all=-0.00071, difference_all_ci95=0.0011)
    assert "cost" not in V.wellness_view(tiny)["tip"]
    assert V.wellness_view(dict(w, all_n=4)) is None
    alone = V.wellness_view(dict(w, reference_n=3))
    assert "Against only" not in alone["text"] and "without the act (" not in alone["tip"]
    # the same verdict against both: one sentence
    same = V.wellness_view(dict(w, difference=0.003))
    assert same["text"].endswith("with or without the act in that time.")
    # the store keeps both sets per act
    t = ST.DetTable.from_dicts(_well_dets(0.002, 0.012, 0.001)[:1])
    assert (int(t["wref_n"][0]), int(t["wall_n"][0])) == (4, 6) and float(t["wall_sum"][0]) == pytest.approx(0.006)
    # small changes keep their digits (the page's two decimals would print 0.0042 as +0)
    assert V._signed(0.0042) == "+0.0042" and V._signed(-3e-5) == "-0.00003" and V._signed(0.1234) == "+0.12"


def test_the_store_says_whether_the_credited_order_is_later_or_earlier():
    from mimo_bridge.tools.observer import store as ST
    rows = []
    for seq, first in ((8, 7), (7, 8), (8, 8)):
        d = _det(100 * seq)
        d["evidence"]["credit"] = dict(provenance="spontaneous", moved=seq != first, measured=True,
                                       order_seq=seq, first_order_seq=first)
        rows.append(d)
    rows.append(_det(900))
    t = ST.DetTable.from_dicts(rows)
    assert t["credit_moved"].tolist() == [1, 2, 0, -1]
    st = TR.evaluate(rows, dict(steps=1000, by_author={"spontaneous": 1000}, windows={}), None, None)
    sel = st["stages"]["selected_intentionally"]
    assert (sel["credited_later"], sel["credited_earlier"], sel["credit_measured"]) == (1, 1, 3)


def test_the_wellness_references_start_at_the_acts_own_place_in_its_order():
    """Wellness moves in a sawtooth inside every 5-step order (measured on H84: -0.012 at the record after
    the latch, +0.019 at the next). An act starting at its order's third record and lasting 3 records
    changes wellness by -0.007 from the sawtooth alone; its references must start at the same place in
    their orders (the same -0.007: no difference), where stretches starting at an order's first record
    read +0.005. The track's first order may have begun before its first record: its records are not
    matched on a guessed place."""
    from mimo_bridge.tools.observer import evidence as EV
    saw = {0: 0.0, 1: -0.012, 2: 0.007, 3: 0.005, 4: 0.002}      # the summed wellness at each place

    class _Lane:
        rows = {s: dict(wellness={"coherence": np.array([saw[s % 5], 0.0])}) for s in range(990, 1130)}

    class _Tr(_WellTrack):
        def __init__(self):
            super().__init__()
            self.order_seq = self.step // 5
    j = EV.Joiner(None, [], [], dict(patch_bodies=[], entities=[]))
    j.lane = _Lane()
    tr = _Tr()
    assert EV.order_phase(tr)[:6].tolist() == [-1, -1, -1, -1, -1, 0]
    act = dict(cap="H05", step0=1062, step1=1065)
    act["evidence"] = dict(wellness=j.wellness(act, 0))
    assert act["evidence"]["wellness"]["total_change"] == pytest.approx(-0.007)
    j.wellness_references([act], 0, tr, all_dets=[act])
    wl = act["evidence"]["wellness"]
    # stretches from 1042 and 1022, the third record of their orders; 1002 lies in the track's first order
    assert wl["all_n"] == 2 and wl["all_sum"] / wl["all_n"] == pytest.approx(-0.007)
    assert wl["reference_n"] == 2 and wl["reference_sum"] / wl["reference_n"] == pytest.approx(-0.007)
