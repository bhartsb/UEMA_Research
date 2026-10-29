# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Round-three review probes for commit 8b498ffc8 (the round-two review's findings R1 to R6 and its test gaps fixed).

Written by the independent round-three review of Oct 5, 2026 (docs/gain_of_function_v4/coordinator/
relaunch_repair_run_round3_review_2026-10-05.md, branch review/relaunch-repair-run-round3-2026-10-05). The real hourly
watcher runs on temporary folders with a simulated clock (the re-check's harness); the relaunch script's main() runs with
every process call replaced (the review's harness, main_under_probe). Nothing here looks at, ends or starts a real
trainer, supervisor or watcher, and nothing is written outside pytest's temporary folder. A probe that asserts the
behaviour a finding asks for and fails on 8b498ffc8 is marked xfail(strict=True) with the finding's id (T1 to T4).

Record times below were read on Oct 5, 2026 from production's run folders (read only): the (model step, seconds after
the trainer's start) of every full stats record, and the start step from each folder's args.json "resume".
"""
import builtins
import json
import math
import time
from pathlib import Path

import pytest

from mimo_bridge.tests.test_h84_hourly_watch import row
from mimo_bridge.tests.test_relaunch_repair_run_recheck import (SEG41C_ARRIVALS, arrivals,
                                                                 watcher_started_before_the_first_record)
from mimo_bridge.tests.test_relaunch_repair_run_review import main_under_probe, record
from mimo_bridge.tests.test_training_preflight import STOP_CRITERIA, _bounded, _repair_setup, _run
from mimo_bridge.tools import training_preflight as TP

# H84_fresh_drivers_20260918_seg41o_1431, the run of checkpoint 481,800 (training is held there): resumed 479,350
SEG41O_START = 479350
SEG41O_TIMES = ((479400, 167.5), (479500, 394.1), (479600, 625.7), (479700, 861.3), (479800, 1090.1),
                (479900, 1326.8), (480000, 1575.0), (480100, 1868.4), (480200, 2109.7), (480300, 2349.6),
                (480400, 2593.0), (480500, 2835.3), (480600, 3097.7), (480700, 3340.5), (480800, 3583.7),
                (480900, 3864.2), (481000, 4194.2), (481100, 4525.1), (481200, 4848.4), (481300, 5172.6),
                (481400, 5493.3), (481500, 5797.9), (481600, 6125.8), (481700, 6461.1), (481800, 6706.0))
# H84_fresh_drivers_20260918_seg41h_0223, in seg41o_1431's resume chain: the fastest pace between two consecutive
# records of the 40 seg41 folders' first readings (0.4703 model steps per second, against seg41o_1431's 0.4413)
SEG41H_START = 472275
SEG41H_TIMES = ((472300, 104.5), (472400, 321.4), (472500, 535.0), (472600, 750.8), (472700, 966.4),
                (472800, 1180.6), (472900, 1397.1), (473000, 1615.3), (473100, 1831.4), (473200, 2045.6))
# H84_fresh_drivers_20260918_seg41b_1851: the field mechanical_contact.5.contact_footprints.5.polygon_tangent.0.0 (a
# position of a list whose length is the number of contacts) reads a number in records 1 and 16 only
SEG41B_START = 429050
SEG41B_TIMES = ((429100, 224.4), (429200, 489.6), (429300, 754.2), (429400, 1015.2), (429500, 1280.7),
                (429600, 1546.0), (429700, 1829.4), (429800, 2113.7), (429900, 2390.3), (430000, 2673.9),
                (430100, 2946.7), (430200, 3239.6), (430300, 3530.7), (430400, 3890.0), (430500, 4157.2),
                (430525, 4307.0))
SEG41B_PRESENT = "1000000000000001"
# H84_fresh_drivers_20260918_seg41s_1146_resume_20261001_213502 (in seg41o_1431's resume chain):
# vestibular.improvement_over_persistence by record (the round-two review's probe of R2)
SEG41S_PRESENT = "00100000000000000001101111111110111111111100"
FIELD_HOLDS = dict(name="probe_field", path="probe.value", op=">", line=1e30, from_run_hour=0, mode="stop")


def start_run(tmp_path, name, start_step, times, end_step, present=None, resume=None):
    """A run folder with full stats records at `times` (each reading probe.value 0.2 where `present` reads 1, and the
    fields of the tests' STOP_CRITERIA in every record); returns its checkpoint ckpt_step_<end_step>."""
    run = tmp_path / "experiments" / name
    ck = run / "checkpoints" / ("ckpt_step_%d" % end_step)
    ck.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({"resume": resume or "C:/x/checkpoints/ckpt_step_%d" % start_step}),
                                   encoding="utf-8")
    lines = []
    for i, (step, elapsed) in enumerate(times):
        r = dict(type="mimo", step=step, elapsed_s=elapsed, timestamp=1_000_000.0 + elapsed,
                 causal_pe_per_modality=dict(vision=0.01), cmd_response_cos_held=0.5)
        if present is None or present[i] == "1":
            r["probe"] = dict(value=0.2)
        lines.append(json.dumps(r) + "\n")
    (run / "replay_training_stats.jsonl").write_text("".join(lines), encoding="utf-8")
    return ck


def curve(start, times):
    return [(elapsed, step - start) for step, elapsed in times]


def steps_at(pts, t):
    """Model steps since the start at `t` seconds after it, linear between records."""
    for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return s0 + (s1 - s0) * (t - t0) / (t1 - t0)
    raise ValueError("outside the records")


def time_at(pts, s):
    for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
        if s0 <= s <= s1:
            return t0 + (t1 - t0) * (s - s0) / (s1 - s0)
    raise ValueError("outside the records")


def read_at(completed):
    """The model step (since a start divisible by 25) at which the trainer reads a stop written after `completed`
    model steps: train_mimo.py polls control.json at the top of its loop when its model step is divisible by 25."""
    return 25 * math.ceil((math.floor(completed) + 1) / 25)


def breaching(step):
    return row(step=step, causal_pe_per_modality=dict(vision=0.1, tactile=.04, proprioception=.005))


# ------------------------------------------------------------------ T1: the first-reading bound on real record times
def test_the_first_reading_at_seg41o_1431s_real_record_times_is_271_model_steps(tmp_path):
    # the commit message's 247 comes from a fixture holding only seg41o_1431's first and last records (its average
    # pace, 0.3671 model steps per second); the real file's fastest interval is records 1 to 2, 100 model steps in
    # 226.6 s (0.4413 per second): 100 + ceil(0.4413 x 330) + 25 = 271
    ck = start_run(tmp_path, "seg41o_like", SEG41O_START, SEG41O_TIMES, 481800)
    assert TP.fastest_rate(TP.stats_records(ck.parents[1])) == pytest.approx(100 / (394.1 - 167.5))
    block = dict(start_checkpoint=str(ck), stop_criteria=STOP_CRITERIA)
    assert TP.first_readings(block, 100) == {"vision_causal_error": 271}
    assert TP.first_reading_problems(dict(block, max_model_steps=271))
    assert TP.first_reading_problems(dict(block, max_model_steps=272)) == []


def test_at_seg41o_1431s_own_pace_the_stop_is_read_21_model_steps_before_the_bound(tmp_path, monkeypatch):
    # the bound's slack at the start run's own pace: at every poll phase tried the trainer reads the stop at 250
    ck = start_run(tmp_path, "seg41o_like", SEG41O_START, SEG41O_TIMES, 481800)
    bound = TP.first_readings(dict(start_checkpoint=str(ck), stop_criteria=STOP_CRITERIA), 100)["vision_causal_error"]
    pts = curve(SEG41O_START, SEG41O_TIMES)
    reads = []
    for delay in range(0, 30, 6):              # the watcher starts `delay` seconds after the trainer
        recs = [(time_at(pts, s) - delay, breaching(481800 + s)) for s in (100, 200, 300)]
        stop, at, why = watcher_started_before_the_first_record(tmp_path / ("d%d" % delay), monkeypatch, recs,
                                                                criteria=STOP_CRITERIA, hours=1)
        assert stop == {"cmd": "snapshot_and_stop"}, why
        reads.append(read_at(steps_at(pts, at + delay)))
    assert set(reads) == {250} and bound - max(reads) == 21


@pytest.mark.xfail(strict=True, reason="T1: first_readings assumes no pace faster than the start run's fastest "
                                       "interval; seg41h_0223, in seg41o_1431's resume chain, ran 0.4703 against 0.4413")
def test_a_bound_accepted_at_seg41o_1431s_times_is_read_before_its_end_at_every_measured_pace(tmp_path, monkeypatch):
    # a repair run from 481,800 bounded at 272 model steps (accepted: the bound reads 271) running at seg41h_0223's
    # measured pace: its records at 100 and 200 model steps come 267.2 s and 481.6 s after its start; when the watcher's
    # look finds the first record 22 to 30 s after it is written, the first check comes after 250 model steps and the
    # trainer would read the stop at 275, after its bound
    ck = start_run(tmp_path, "seg41o_like", SEG41O_START, SEG41O_TIMES, 481800)
    bound = 272
    assert TP.first_reading_problems(dict(start_checkpoint=str(ck), stop_criteria=STOP_CRITERIA,
                                          max_model_steps=bound)) == []
    pts = curve(SEG41H_START, SEG41H_TIMES)
    reads = []
    for delay in range(0, 30, 2):
        recs = [(time_at(pts, s) - delay, breaching(481800 + s)) for s in (100, 200)]
        stop, at, why = watcher_started_before_the_first_record(tmp_path / ("d%d" % delay), monkeypatch, recs,
                                                                criteria=STOP_CRITERIA, hours=1)
        assert stop == {"cmd": "snapshot_and_stop"}, why
        reads.append(read_at(steps_at(pts, at + delay)))
    # the trainer's loop runs while its model step is under --max-steps, so it reads a stop only at a step below the bound
    assert max(reads) < bound, reads


# ------------------------------------------------------------------ T2: the absence check from the third record
def _seg41b_run(tmp_path):
    return start_run(tmp_path, "seg41b_like", SEG41B_START, SEG41B_TIMES, 430525, present=SEG41B_PRESENT)


def test_a_field_the_check_accepts_on_the_start_runs_records_is_stopped_by_the_watcher_on_the_same_records(
        tmp_path, monkeypatch):
    # the check counts the absence from the start run's third record (754.2 s) to its next reading (4,307.0 s): 3,553 s,
    # under the hour; the watcher counts its hour from the first record (224.4 s, where the field read a number) and,
    # at its check of 4,140 s (newest record 3,890.0 s), finds the field in no record of the last hour
    ck = _seg41b_run(tmp_path)
    assert TP.stop_field_problems(dict(start_checkpoint=str(ck), stop_criteria=[FIELD_HOLDS])) == []
    recs = [(elapsed, dict(row(step=step), **({"probe": dict(value=0.2)} if SEG41B_PRESENT[i] == "1" else {})))
            for i, (step, elapsed) in enumerate(SEG41B_TIMES)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "w", monkeypatch, recs, criteria=[FIELD_HOLDS],
                                                            hours=2)
    assert stop == {"cmd": "snapshot_and_stop"} and "probe.value is absent" in why and at == 4140, (at, why)


@pytest.mark.xfail(strict=True, reason="T2: stop_field_problems counts absence from the start run's third record; the "
                                       "watcher's hour of grace counts from its first")
def test_a_field_the_watcher_would_stop_on_in_the_start_runs_own_records_is_refused(tmp_path):
    ck = _seg41b_run(tmp_path)
    assert TP.stop_field_problems(dict(start_checkpoint=str(ck), stop_criteria=[FIELD_HOLDS]))


# ------------------------------------------------------------------ T3: absence in the start run's resume chain
def _chain(tmp_path):
    """seg41o_1431's case: the start run writes the field in every record; a run of its resume chain
    (seg41s_1146_resume_20261001_213502) left it absent from its 4th to its 19th record."""
    times = [(451500 + 100 * i, at) for i, at in enumerate(arrivals(first=(285, 513, 739),
                                                                     until=285 + 230 * (len(SEG41S_PRESENT) - 1)))]
    parent = start_run(tmp_path, "seg41s_like", 451425, times[:len(SEG41S_PRESENT)], times[len(SEG41S_PRESENT) - 1][0],
                       present=SEG41S_PRESENT)
    return start_run(tmp_path, "seg41o_like", SEG41O_START, SEG41O_TIMES, 481800, resume=str(parent).replace("\\", "/"))


def test_a_field_absent_for_an_hour_in_the_start_runs_resume_chain_is_accepted(tmp_path, monkeypatch):
    ck = _chain(tmp_path)
    assert TP.stop_field_problems(dict(start_checkpoint=str(ck), stop_criteria=[FIELD_HOLDS])) == []
    # a repair run whose records repeat the chain's pattern is stopped, its line holding (the round-two review's probe)
    recs = []
    for i, at in enumerate(arrivals(first=(285, 513, 739), until=285 + 230 * (len(SEG41S_PRESENT) - 1))):
        recs.append((at, dict(row(step=481900 + 100 * i), **({"probe": dict(value=0.2)} if SEG41S_PRESENT[i] == "1"
                                                              else {}))))
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "w", monkeypatch, recs, criteria=[FIELD_HOLDS],
                                                            hours=3)
    assert stop == {"cmd": "snapshot_and_stop"} and "is absent" in why and at == 4500, (at, why)


@pytest.mark.xfail(strict=True, reason="T3: stop_field_problems reads the start run alone; seg41o_1431's resume chain "
                                       "holds the hour-long absences its records do not show")
def test_a_field_absent_for_an_hour_in_the_start_runs_resume_chain_is_refused(tmp_path):
    ck = _chain(tmp_path)
    assert TP.stop_field_problems(dict(start_checkpoint=str(ck), stop_criteria=[FIELD_HOLDS]))


# ------------------------------------------------------------------ T4: a failed attach spends the repair run
def _failed_attach(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "silent", supervisor_reports=False)
    with pytest.raises(SystemExit, match="did not report the new run"):
        module.main()
    run = p.launched["run"]
    assert json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    # the supervisor did not report for 120 s; the trainer, training since its load (31.3 s in seg41o_1431), reads the
    # request at its next model step divisible by 25, saves, and writes stage_done.json with that step at its end
    # (train_mimo.main -> training_continuation.write_stage_done): here 50 model steps past the start, no full stats
    # record written (the first comes at 100)
    (run / "stage_done.json").write_text(json.dumps({"steps": 1050, "ok": True}), encoding="utf-8")
    return p, run, TP.apply_sets(p.launched["cmd"], [("--max-steps", "3000")])


def test_a_failed_attach_after_the_trainers_first_model_steps_spends_the_repair_run(tmp_path, monkeypatch):
    p, run, again = _failed_attach(tmp_path, monkeypatch)
    rr = record(p)
    assert rr["launched_at"] is None and TP.repair_launch(rr)[2] == str(Path(run).resolve())
    assert TP.launch_plan(again, root=p.root, src=p.src)["kind"] == "normal"                   # no retry
    assert "never normal training" in TP.launch_refusal(again, root=p.root, src=p.src, registry=p.reg)
    with pytest.raises(TP.PreflightError, match="needs its own record"):
        TP.set_repair_run(p.fp, rr, "tester", root=p.root, registry=p.reg)
    with pytest.raises(TP.PreflightError, match="exists"):                                     # one record per fingerprint
        TP.new_record(TP.apply_sets(again, [("--max-steps", "0")]), "a retry", "tester", root=p.root, src=p.src)


@pytest.mark.xfail(strict=True, reason="T4: an attempt the relaunch stopped before its first full stats record, for want "
                                       "of a supervisor, counts as the record's one repair run")
def test_a_failed_attach_before_the_first_full_stats_record_leaves_the_repair_run_to_retry(tmp_path, monkeypatch):
    p, run, again = _failed_attach(tmp_path, monkeypatch)
    assert TP.launch_plan(again, root=p.root, src=p.src)["kind"] == "repair_run"


# ------------------------------------------------------------------ question 4: the repair path of main() end to end
def test_main_notes_the_attempt_before_the_trainer_starts_and_marks_the_run_at_its_first_record(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")
    launch0, seen = module.L.launch_trainer, {}

    def launch(cmd, run):
        seen.update(rr=record(p), control=json.loads((p.saved / "supervisor_control.json").read_text("utf-8")))
        return launch0(cmd, run)
    monkeypatch.setattr(module.L, "launch_trainer", launch)
    module.main()
    run = p.launched["run"]
    assert [a["run"] for a in seen["rr"]["attempts"]] == [str(Path(run).resolve())]
    assert seen["rr"]["launched_at"] is None and seen["control"]["repair_run"] == str(run)
    assert len(p.started) == 1 and p.started[0]["at"] == p.launched["at"]                  # attached at the start
    assert record(p)["launched_at"] and record(p)["run"] == str(Path(run).resolve())


def test_a_failed_note_of_the_attempt_ends_the_script_before_any_trainer_starts(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")

    def refuse(*a, **k):
        raise OSError("the preflight record could not be written")
    monkeypatch.setattr(TP, "note_attempt", refuse)
    with pytest.raises(OSError, match="could not be written"):
        module.main()
    assert not p.launched and not p.started
    assert TP.repair_launch(record(p))[0] is None                                   # nothing trained: unused
    assert TP.launch_plan(p.asked[-1], root=p.root, src=p.src)["kind"] == "repair_run"


def test_a_trainer_that_never_starts_leaves_its_noted_attempt_unused(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")

    def fail(cmd, run):
        raise OSError("the trainer could not be started")
    monkeypatch.setattr(module.L, "launch_trainer", fail)
    with pytest.raises(OSError, match="could not be started"):
        module.main()
    rr = record(p)
    assert len(rr["attempts"]) == 1 and TP.repair_launch(rr)[0] is None and not p.started
    assert TP.launch_plan(p.asked[-1], root=p.root, src=p.src)["kind"] == "repair_run"


# ------------------------------------------------------------------ question 2: the watcher's new rules
def test_one_failed_read_at_the_first_check_moves_the_first_judgement_one_check_later(tmp_path, monkeypatch):
    # first_readings counts the first check (300 s after the look that finds the first record, 570 s here); one failed
    # read there moves the first judgement of a breached line to the next check, 870 s
    real_open, failed = builtins.open, []

    def open_once_refused(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if str(file).endswith("declared_criteria.json") and mode == "r" and not failed:
            failed.append(str(file))
            raise PermissionError(13, "the file is held by another process", str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", open_once_refused)
    recs = [(at, breaching(447400 + 100 * i)) for i, at in enumerate(SEG41C_ARRIVALS)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, recs, criteria=STOP_CRITERIA,
                                                            hours=1)
    assert failed and stop == {"cmd": "snapshot_and_stop"} and at == 270 + 600, (at, why)


def test_a_non_finite_stop_field_stops_the_run_on_either_side_of_its_line(tmp_path, monkeypatch):
    # the round-two review's R3 repair as named: Infinity in the newest record stops the run at once, also for a line
    # it holds (a stop when the error falls UNDER 0.001; Infinity is over it)
    under = dict(name="error_falls_under", path="causal_pe_per_modality.vision", op="<", line=0.001, from_run_hour=0,
                 mode="stop")
    recs = [(at, row(step=447400 + 100 * i, causal_pe_per_modality=dict(vision=float("inf"), tactile=.04,
                                                                       proprioception=.005)))
            for i, at in enumerate(SEG41C_ARRIVALS)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, recs, criteria=[under], hours=1)
    assert stop == {"cmd": "snapshot_and_stop"} and at == 270 + 300 and "reads inf in the newest record" in why


# ------------------------------------------------------------------ test gaps: mutations no commit test caught
def test_after_a_repair_run_used_through_its_attempt_normal_training_continues_only_the_repaired_model(tmp_path):
    # mutation M14 (launch_refusal's normal branch reads only the marking): a record used through its attempt (the
    # script ended before marking it) let gates judged again allow normal training from the start checkpoint, whose
    # weights the repair run never changed (the review's finding F9)
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    attempt = _run(tmp_path, "run_attempt", (3000,), resume=start)
    TP.note_attempt(fp, str(attempt), "tester", root=root)
    (attempt / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n",
                                                         encoding="utf-8")
    time.sleep(0.05)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "PASS", "measured again on the repair run's end checkpoint", ["row_%db" % n], "tester",
                    root=root)
    why = TP.launch_refusal(cmd, root=root, src=src, registry=reg)                  # the start: refused
    assert why and "ran its repair run" in why
    end = attempt / "checkpoints" / "ckpt_step_3000"
    TP.add_state(fp, str(end), "gates 4, 5 and 7 judged again on it", "tester", root=root)
    later = list(cmd)
    later[later.index("--resume") + 1] = str(end)
    assert TP.launch_refusal(later, root=root, src=src, registry=reg) is None


def test_an_attempt_noted_with_a_relative_folder_is_read_from_any_working_folder(tmp_path, monkeypatch):
    # mutation M23 (note_attempt stores the folder as given): the attempt read from another working folder found no
    # training, and the used record read as unused
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    attempt = tmp_path / "experiments" / "run_attempt"
    attempt.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    TP.note_attempt(fp, "experiments/run_attempt", "tester", root=root)
    (attempt / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n",
                                                         encoding="utf-8")
    monkeypatch.chdir(tmp_path / "experiments")
    assert TP.launch_plan(_bounded(cmd, 3000), root=root, src=src)["kind"] == "normal"


def test_an_attempt_needs_its_existing_run_folder(tmp_path):
    # mutation M24 (note_attempt accepts a missing folder)
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    with pytest.raises(TP.PreflightError, match="does not exist"):
        TP.note_attempt(fp, str(tmp_path / "experiments" / "never_made"), "tester", root=root)
    assert not TP._load(TP.record_path(fp, root))["repair_run"].get("attempts")


def _criteria_reads_failing(monkeypatch, failing):
    """Make the given reads (1 = the first) of declared_criteria.json raise PermissionError; returns the read count."""
    real_open, reads = builtins.open, []

    def open_some_refused(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if str(file).endswith("declared_criteria.json") and mode == "r":
            reads.append(str(file))
            if len(reads) in failing:
                raise PermissionError(13, "the file is held by another process", str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", open_some_refused)
    return reads


def test_two_failed_reads_with_a_good_read_between_stop_nothing(tmp_path, monkeypatch):
    # mutation M37 (the count not reset by a readable read): failed reads at the 1st and 3rd 5-minute checks, the line
    # holding, stopped the run at the 3rd (1,170 s)
    reads = _criteria_reads_failing(monkeypatch, failing=(1, 3))
    recs = [(at, row(step=447400 + 100 * i)) for i, at in enumerate(arrivals(until=3600))]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, recs, criteria=STOP_CRITERIA,
                                                            hours=1)
    assert len(reads) >= 4 and stop is None, (at, why)


def test_one_failed_read_at_the_hourly_sweep_stops_nothing(tmp_path, monkeypatch):
    # mutation M40 (the sweep stops at once on an unreadable file, the rule of 960ccfb1c): the 11th read is the first
    # hourly sweep's (after the first hour's 10 checks); the 5-minute checks before and after it read the file
    reads = _criteria_reads_failing(monkeypatch, failing=(11,))
    recs = [(at, row(step=447400 + 100 * i)) for i, at in enumerate(arrivals(until=7200))]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, recs, criteria=STOP_CRITERIA,
                                                            hours=2)
    assert len(reads) > 11 and stop is None, (at, why)


def test_the_non_finite_stop_reads_the_newest_record(tmp_path, monkeypatch):
    # mutation M43 (the window's oldest record read instead): the first record's Infinity, followed by finite readings
    # under the line, stopped the run; R3's rule as named reads the newest record, and the window mean leaves the
    # Infinity out
    recs = [(at, row(step=447400 + 100 * i, causal_pe_per_modality=dict(vision=float("inf") if i == 0 else 0.01,
                                                                       tactile=.04, proprioception=.005)))
            for i, at in enumerate(arrivals(until=3600))]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, recs, criteria=STOP_CRITERIA,
                                                            hours=1)
    assert stop is None, (at, why)
