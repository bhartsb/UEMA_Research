# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The motor fixes' kill-line watcher (docs/gain_of_function_v4/motor_fixes_watch.py) on synthetic
stats records and observer chunks, in temporary folders only: it writes the stop when, and only when,
a declared line is crossed over full windows, never while the ability scheduler holds training, never
twice for one crossing across a restart, and it judges the eye-stop line over the joined records of
the deploy run and its continuations (reviews of 2026-09-25: the first branch's findings 1 and 4, the
deploy branch's findings 1, 2, 6, 7 and 9).
"""
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

DOCS = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4"


def _load():
    spec = importlib.util.spec_from_file_location("motor_fixes_watch", DOCS / "motor_fixes_watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load()
MB = W.MB
N = 410_000


def rec(step, body=(0.40, 0.70), gaze=(2.0, 3.4), causal=0.0165, cos=0.406):
    """One stats window as the trainer writes it (the fields the lines read)."""
    return {"type": "mimo", "step": step, "timestamp": 1000.0 + step,
            "efference": {"body_cm_err_next": body[0], "body_cm_shuffled_next": body[1],
                          "gaze_deg_err": [gaze[0], None, None], "gaze_deg_shuffled": [gaze[1], None, None]},
            "causal_pe": causal, "cmd_response_cos_held": cos}


def make_run(root, name, records, switch=None, parent=None, checkpoint=None):
    run = Path(root) / name
    run.mkdir(parents=True)
    lines = []
    if switch is not None:
        lines.append(json.dumps({"type": "motor_interface_change", "step": switch,
                                 "changes": {"executed_fields_only": [False, True]}}))
    lines += [json.dumps(r) for r in records]
    (run / "replay_training_stats.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if parent is not None:
        (run / "continuation_receipt.json").write_text(json.dumps(
            {"parent_run": str(parent), "checkpoint": str(checkpoint)}), encoding="utf-8")
    return run


def append(run, records):
    with open(Path(run) / "replay_training_stats.jsonl", "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def saved_for(tmp_path, active, scheduler=None, control=None):
    saved = tmp_path / "saved"
    saved.mkdir(exist_ok=True)
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(active)}),
                                                      encoding="utf-8")
    if scheduler is not None:
        (saved / "ability_rechecks").mkdir(exist_ok=True)
        (saved / "ability_rechecks" / "status.json").write_text(json.dumps(scheduler), encoding="utf-8")
    if control is not None:
        (saved / "supervisor_control.json").write_text(json.dumps(control), encoding="utf-8")
    return saved


def windows(first, n, **kw):
    return [rec(first + 100 * k, **kw) for k in range(n)]


def ckpt(step):
    return "C:/x/checkpoints/ckpt_step_%d" % step


BAD = dict(body=(0.70, 0.60))                     # the own-order error over the other-order error


def test_the_lines_are_the_ones_the_plan_declares():
    assert W.CONSECUTIVE == 3 and W.REPORT_EVERY == 100
    assert W.CAUSAL_PE_LINE == 0.031 and W.FOLLOW_LINE == 0.335
    assert W.BODY_LINE_CM == 0.0 and W.GAZE_LINE_DEG == 0.0
    assert W.EYE_STOP_LINE_PER_100 == 5.0 and W.EYE_BLOCK == 1700 and MB.STOP_BAND_DEG == 2.0
    assert W.REPLAY_TURNOVER_STEPS == 6000 and W.CHUNK_STEPS == 100
    assert W.SCHEDULER_HOLDS == ("preparing", "saving", "ready", "held for review")


def test_three_full_windows_in_a_row_write_the_stop_and_the_first_window_is_not_judged(tmp_path):
    exp = tmp_path / "exp"
    # windows 1-3 cross, but window 1 is the launch's own: only 2 judged crossings in a row, no stop
    run = make_run(exp, "deploy", windows(N + 100, 3, **BAD) + windows(N + 400, 2), switch=N)
    saved = saved_for(tmp_path, run)
    why, state = W.watch_once(run, saved)
    assert why is None and not (run / "control.json").exists()
    assert state["windows"]["runs"][0]["judged"] == 4
    # a third judged crossing in a row: the stop
    run2 = make_run(exp, "deploy2", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved2 = saved_for(tmp_path, run2)
    why, state = W.watch_once(run2, saved2)
    assert why and "body head" in why and "3 full windows" in why
    raw = (run2 / "control.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 without a byte-order mark"
    assert json.loads(raw.decode("utf-8"))["cmd"] == "snapshot_and_stop"
    assert "auto_resume" not in json.loads(raw.decode("utf-8"))
    stop = json.loads((run2 / "stop_reason.json").read_text(encoding="utf-8"))
    assert stop["resume_policy"] == "repair_required"
    sup = json.loads((saved2 / "supervisor_control.json").read_text(encoding="utf-8"))
    assert sup["cmd"] == "pause"
    # the supervisor would not relaunch this run (its own reader of the stop)
    from mimo_bridge.tools.training_continuation import pause_reason
    assert pause_reason(tmp_path / "no_saved_here", run2) is not None


def test_the_short_record_written_at_a_stop_neither_counts_nor_breaks_a_run(tmp_path):
    """Review of the deploy branch, finding 2, its failing input: two full windows past the body head's
    line, then the 30-step record the trainer writes when the ability scheduler's window stops the run.
    Judged as a window, it completed a crossing and the stop landed inside the scheduler's window."""
    exp = tmp_path / "exp"
    recs = windows(N + 100, 1) + windows(N + 200, 2, **BAD) + [rec(N + 330, **BAD)]
    run = make_run(exp, "deploy", recs, switch=N)
    why, state = W.watch_once(run, saved_for(tmp_path, run))
    assert why is None and not (run / "control.json").exists()
    assert state["windows"]["runs"][0]["short"] == 1 and state["windows"]["runs"][0]["judged"] == 2
    # a short record on the safe side between full windows past the line does not break their run
    # (rule semantics: the trainer writes a short record only at a stop, so none follows it in a launch)
    run2 = make_run(exp, "deploy2", windows(N + 100, 1) + windows(N + 200, 2, **BAD)
                    + [rec(N + 330)] + [rec(N + 430, **BAD)], switch=N)
    why, state = W.watch_once(run2, saved_for(tmp_path, run2))
    assert why is not None and state["windows"]["runs"][0]["short"] == 1


def test_a_launch_that_resumed_between_reports_starts_with_a_short_record(tmp_path):
    """A continuation resumed from a checkpoint at step C writes its first record at the next multiple
    of 100; windows after it are full."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 2), switch=N)
    cont = make_run(exp, "deploy_resume_1", [rec(N + 300, **BAD)] + windows(N + 400, 3, **BAD),
                    parent=deploy, checkpoint=ckpt(N + 250))
    why, state = W.watch_once(deploy, saved_for(tmp_path, cont))
    assert why and "deploy_resume_1" in why
    assert state["spans"] == {"deploy": [N, N + 250], "deploy_resume_1": [N + 250, None]}


def test_a_safe_window_restarts_the_count(tmp_path):
    recs = windows(N + 100, 1) + windows(N + 200, 2, **BAD) + windows(N + 400, 1) + windows(N + 500, 2, **BAD)
    run = make_run(tmp_path / "exp", "deploy", recs, switch=N)
    why, state = W.watch_once(run, saved_for(tmp_path, run))
    assert why is None and state["windows"]["runs"][0]["counts"]["body head"] == 2


def test_a_full_window_without_the_value_breaks_the_run(tmp_path):
    hole = rec(N + 400)
    hole["efference"]["body_cm_shuffled_next"] = None
    recs = windows(N + 100, 1) + windows(N + 200, 2, **BAD) + [hole] + windows(N + 500, 2, **BAD)
    run = make_run(tmp_path / "exp", "deploy", recs, switch=N)
    assert W.watch_once(run, saved_for(tmp_path, run))[0] is None


@pytest.mark.parametrize("kw,line", [(dict(gaze=(3.0, 2.9)), "gaze head"),
                                     (dict(gaze=(3.0, 3.0)), "gaze head"),       # at the line crosses
                                     (dict(causal=0.0311), "causal error"),
                                     (dict(cos=0.334), "following")])
def test_each_line_stops_on_three_windows(tmp_path, kw, line):
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 1) + windows(N + 200, 3, **kw), switch=N)
    why, _ = W.watch_once(run, saved_for(tmp_path, run))
    assert why and line in why


@pytest.mark.parametrize("kw", [dict(causal=0.031), dict(cos=0.335), dict(body=(0.5, 0.5001))])
def test_values_exactly_on_the_safe_side_do_not_stop(tmp_path, kw):
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 1) + windows(N + 200, 5, **kw), switch=N)
    assert W.watch_once(run, saved_for(tmp_path, run))[0] is None


def test_windows_before_the_switch_are_not_judged(tmp_path):
    bad = dict(causal=0.05)
    run = make_run(tmp_path / "exp", "deploy", windows(N - 500, 5, **bad) + windows(N + 100, 2), switch=N)
    why, state = W.watch_once(run, saved_for(tmp_path, run))
    assert why is None and state["after_step"] == N


def test_a_continuation_restarts_the_count_and_the_stop_lands_in_the_active_run(tmp_path):
    exp = tmp_path / "exp"
    bad = dict(causal=0.05)
    # the scheduler stopped the deploy run at N + 300 (its checkpoint): the deploy run owns its records
    # up to and including N + 300
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 2, **bad), switch=N)
    cont = make_run(exp, "deploy_resume_1", windows(N + 400, 3, **bad), parent=deploy, checkpoint=ckpt(N + 300))
    assert [p.name for p in W.watched_runs(deploy)] == ["deploy", "deploy_resume_1"]
    # 2 judged crossings in the deploy run, the continuation's first window skipped, 2 more there: the
    # counts restart in each launch, so no stop
    why, state = W.watch_once(deploy, saved_for(tmp_path, cont))
    assert why is None
    append(cont, [rec(N + 700, **bad)])
    why, _ = W.watch_once(deploy, saved_for(tmp_path, cont))
    assert why and "deploy_resume_1" in why
    assert (cont / "control.json").exists() and not (deploy / "control.json").exists()


def test_the_steps_a_crash_resume_abandoned_are_not_judged(tmp_path):
    """The deploy run crashed at N + 900 after its checkpoint at N + 500; the continuation resumed from
    that checkpoint, so the deploy run's windows after N + 500 belong to an abandoned future."""
    exp = tmp_path / "exp"
    bad = dict(causal=0.05)
    deploy = make_run(exp, "deploy", windows(N + 100, 5) + windows(N + 600, 4, **bad), switch=N)
    cont = make_run(exp, "deploy_resume_1", windows(N + 600, 3), parent=deploy, checkpoint=ckpt(N + 500))
    why, state = W.watch_once(deploy, saved_for(tmp_path, cont))
    assert why is None
    assert state["windows"]["runs"][0]["last_step"] == N + 500


def test_nothing_is_written_while_the_scheduler_holds_training_and_the_stop_waits(tmp_path):
    """Review finding 2: the watcher's pause replaced the scheduler's own control, cancelling the
    window's comparisons, and during a review hold it recorded the failed round as reviewed. The stop
    now waits for every hold, and is written once none holds."""
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    for hold in ({"comparison_window": {"state": "held for review"}},
                 {"comparison_window": {"state": "saving"}},
                 "unreadable"):
        saved = saved_for(tmp_path, run, scheduler=hold if isinstance(hold, dict) else {})
        if hold == "unreadable":
            # the scheduler's status caught mid-replace (the second review, finding 6: it read as no hold)
            (saved / "ability_rechecks" / "status.json").write_bytes(b'{"comparison_window": {"sta')
        why, state = W.watch_once(run, saved)
        assert why is None and state["pending_stop"]["reason"]
        assert not (run / "control.json").exists() and not (run / "stop_reason.json").exists()
        assert not (saved / "supervisor_control.json").exists()
    assert "unreadable" in state["pending_stop"]["waiting_for"]
    # a supervisor pause already in force (a coordinated pause, another watcher's stop): wait too
    saved = saved_for(tmp_path, run, scheduler={"comparison_window": {"state": "released"}},
                      control={"cmd": "pause", "reason": "Coordinated evaluation pause: jobs"})
    why, state = W.watch_once(run, saved)
    assert why is None and "supervisor pause" in state["pending_stop"]["waiting_for"]
    # a control.json no trainer has consumed yet: wait
    (saved / "supervisor_control.json").write_text(json.dumps({"cmd": "run"}), encoding="utf-8")
    (run / "control.json").write_text(json.dumps({"cmd": "snapshot_and_stop", "auto_resume": True}),
                                      encoding="utf-8")
    why, state = W.watch_once(run, saved)
    assert why is None and "control.json" in state["pending_stop"]["waiting_for"]
    (run / "control.json").unlink()
    # nothing holds: the stop is written
    why, state = W.watch_once(run, saved)
    assert why and json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"


def test_a_restart_after_a_written_stop_does_not_stop_the_continuation_on_the_old_crossing(tmp_path):
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    why, _ = W.watch_once(deploy, saved, memory=W.Memory())
    assert why
    # the operator continues from the stop's checkpoint (lifting the pause the stop wrote); a watcher is
    # started again for the deploy
    cont = make_run(exp, "deploy_resume_1", windows(N + 500, 3), parent=deploy, checkpoint=ckpt(N + 400))
    saved = saved_for(tmp_path, cont, control={"cmd": "run"})
    again = W.Memory(W.load_state(deploy))
    assert again.judged_after() == N + 400 and again.judged_until() == {"deploy": N + 400}
    why, state = W.watch_once(deploy, saved, memory=again)
    assert why is None and state["phase"] == "released", "the operator lifted the stop's pause"
    why, state = W.watch_once(deploy, saved, memory=again)
    assert why is None and not (cont / "control.json").exists()
    # --fresh (no memory): the old crossing is judged again, which is what the memory prevents
    why, _ = W.watch_once(deploy, saved, memory=W.Memory(fresh=True))
    assert why


def test_a_relaunch_by_hand_after_a_stop_is_followed_from_its_args(tmp_path):
    """The supervisor never relaunches a run whose stop requires repair, so training continues by a
    relaunch by hand, which writes no continuation receipt: its args.json --resume names the deploy
    run's checkpoint, and the watcher follows it with the same --deploy-run."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    why, _ = W.watch_once(deploy, saved, memory=W.Memory())
    assert why
    # the checkpoint named by path on C:, and a second relaunch named by a moved checkpoint (another
    # drive, the same run folder name)
    hand = make_run(exp, "deploy_by_hand", windows(N + 500, 3))
    (hand / "args.json").write_text(json.dumps({"resume": str(deploy / "checkpoints" / ("ckpt_step_%d" % (N + 400)))}),
                                    encoding="utf-8")
    other = make_run(exp, "unrelated", windows(N + 500, 4, **BAD))
    (other / "args.json").write_text(json.dumps({"resume": str(exp / "pre" / "checkpoints" / "ckpt_step_1")}),
                                     encoding="utf-8")
    assert [p.name for p in W.watched_runs(deploy)] == ["deploy", "deploy_by_hand"]
    assert W.first_step_of(hand, deploy, N) == N + 400
    moved = make_run(exp, "deploy_by_hand_2", windows(N + 900, 2))
    (moved / "args.json").write_text(json.dumps({"resume": "D:/archive/deploy_by_hand/checkpoints/ckpt_step_%d" % (N + 800)}),
                                     encoding="utf-8")
    assert [p.name for p in W.watched_runs(deploy)] == ["deploy", "deploy_by_hand", "deploy_by_hand_2"]
    again = W.Memory(W.load_state(deploy))
    why, state = W.watch_once(deploy, saved_for(tmp_path, moved, control={"cmd": "run"}), memory=again)
    assert why is None and state["phase"] == "released", "the operator lifted the stop's pause"
    why, state = W.watch_once(deploy, saved_for(tmp_path, moved, control={"cmd": "run"}), memory=again)
    assert why is None, why
    assert state["spans"]["deploy_by_hand"] == [N + 400, N + 800]
    # the relaunch's own crossing is a stop, in the active run
    append(moved, windows(N + 1100, 3, **BAD))
    why, _ = W.watch_once(deploy, saved_for(tmp_path, moved, control={"cmd": "run"}), memory=again)
    assert why and "deploy_by_hand_2" in why and (moved / "control.json").exists()


def test_the_stats_reader_keeps_eight_numbers_a_record(tmp_path):
    """Review finding 7: a parsed H84 record is 0.44 MB; the watcher keeps what the lines read."""
    big = rec(N + 100)
    big["efference"]["liveness"] = {"x%d" % i: [0.1] * 50 for i in range(200)}
    big["mechanical_contact"] = [{"history": list(range(1000))}] * 8
    run = make_run(tmp_path / "exp", "deploy", [big], switch=N)
    recs = W.StatsReader().read(run)
    m = [r for r in recs if r["type"] == "mimo"][0]
    assert set(m) == {"_compact", "type", "step", "body_err", "body_shuf", "gaze_err", "gaze_shuf",
                      "causal_pe", "cos_held"}
    assert W.line_values(m) == W.line_values(big)


def test_the_stats_reader_leaves_a_partly_written_line(tmp_path):
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 2), switch=N)
    with open(run / "replay_training_stats.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec(N + 300))[:40])
    reader = W.StatsReader()
    assert len([r for r in reader.read(run) if r["type"] == "mimo"]) == 2
    with open(run / "replay_training_stats.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec(N + 300))[40:] + "\n")
    assert len([r for r in reader.read(run) if r["type"] == "mimo"]) == 3


def test_nothing_is_written_when_the_active_run_is_not_watched(tmp_path):
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, causal=0.05), switch=N)
    other = make_run(exp, "another_lineage", windows(N, 2))
    saved = saved_for(tmp_path, other)
    why, state = W.watch_once(run, saved)
    assert why is None and state["pending_stop"] and "not one of the watched runs" in state["status"]
    assert not (run / "control.json").exists() and not (other / "control.json").exists()
    assert not (saved / "supervisor_control.json").exists()


def test_without_a_switch_record_nothing_is_judged(tmp_path):
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 5, causal=0.05))
    why, state = W.watch_once(run, saved_for(tmp_path, run))
    assert why is None and state["after_step"] is None and "no switch recorded" in state["status"]
    assert W.main(["--deploy-run", str(run), "--saved", str(tmp_path / "saved"), "--once"]) == 2
    # N given by hand: judged
    assert W.main(["--deploy-run", str(run), "--saved", str(tmp_path / "saved"), "--once",
                   "--after-step", str(N)]) == 1


def test_a_failed_poll_is_said_and_does_not_end_the_watch(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 2), switch=N)
    calls = []
    real = W.watch_once

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("a chunk being written")
        return real(*a, **k)
    monkeypatch.setattr(W, "watch_once", flaky)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    # --once: the failed poll is reported with its own exit code
    assert W.main(["--deploy-run", str(run), "--saved", str(tmp_path / "saved"), "--once"]) == 3
    assert "POLL FAILED" in capsys.readouterr().out
    # a longer watch goes on to the next poll after the failure
    calls.clear()
    assert W.main(["--deploy-run", str(run), "--saved", str(tmp_path / "saved"), "--hours", "1e-9"]) == 0
    assert len(calls) == 2
    assert "ended" in capsys.readouterr().out


def test_ensure_starts_a_watcher_only_when_none_holds_the_lock_and_says_a_waiting_stop(tmp_path):
    """The second review, findings 3 and 6: --ensure starts a watcher only when no process holds the
    deploy run's lock (the state file's heartbeat alone let two run), names a watcher that holds it
    without polling, and prints a waiting or unconfirmed stop first."""
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 2), switch=N)
    started = []

    def start(argv, deploy_run):
        started.append(list(argv))
        return 4242
    # no watcher holds the lock: start
    got, msg = W.ensure(["--deploy-run", str(run)], run, 60.0, held=lambda r: False, start=start)
    assert got and started[-1] == ["--deploy-run", str(run)] and "4242" in msg
    # the lock is held and the holder polled 30 s ago: leave it
    W._write_state(run, {"pid": 7, "time": W.time.time() - 30})
    got, msg = W.ensure([], run, 60.0, held=lambda r: True, start=start)
    assert not got and msg.startswith("alive")
    # held, but silent for more than 5 polls: named, and not replaced while it lives
    W._write_state(run, {"pid": 7, "time": W.time.time() - 400})
    got, msg = W.ensure([], run, 60.0, held=lambda r: True, start=start)
    assert not got and "ALIVE BUT SILENT" in msg and "7" in msg
    # a waiting stop is said first, with what it waits for
    W._write_state(run, {"pid": 7, "time": W.time.time() - 30, "pending_stop": {
        "reason": "motor fixes kill line crossed: body head", "waiting_for": "a supervisor pause is in force (x)",
        "since": W.time.time() - 600}})
    _, msg = W.ensure([], run, 60.0, held=lambda r: True, start=start)
    assert msg.splitlines()[0].startswith("STOP WAITING") and "a supervisor pause is in force" in msg
    # the real lock: held by this process, so a second holder is refused
    lock = W.acquire_lock(run)
    try:
        assert W.lock_held(run)
        assert W.main(["--deploy-run", str(run), "--saved", str(tmp_path / "saved"), "--once"]) == 4
    finally:
        lock.close()
    assert not W.lock_held(run)
    # --ensure passes every other argument on to the watcher it starts
    seen = []
    orig = W.ensure
    W.ensure = lambda child, deploy_run, poll, **k: (seen.append(child) or True, "started")
    try:
        assert W.main(["--deploy-run", str(run), "--ensure", "--baseline-run", "X", "--poll", "30"]) == 0
    finally:
        W.ensure = orig
    assert seen[0] == ["--deploy-run", str(run), "--baseline-run", "X", "--poll", "30"]


def test_the_readings_name_their_segments_and_read_n_from_the_switch_record(tmp_path, monkeypatch):
    """Review of the first branch, findings 5 and 6: a deploy run in a new segment is read, and N comes
    from the deploy's own record - never from a record of a revert back to the default."""
    lin = MB.LINEAGE
    assert MB.parse_segments("36,37, 40") == {36, 37, 40}
    assert MB.parse_segments("run:C:/x/%s41_motor_fixes" % lin) == {41}
    assert MB.parse_segments("40,run:%s41_motor_fixes_resume_20260926_010203" % lin) == {40, 41}
    with pytest.raises(SystemExit):
        MB.parse_segments("run:C:/x/some_other_run")
    exp = tmp_path / "experiments"
    f4 = exp / (lin + "41_motor_fixes")
    f1 = exp / (lin + "41_motor_fixes_resume_20260926_010203")
    back = exp / (lin + "41_motor_fixes_resume_20260926_050000")
    for folder, step, changes, ts in ((f4, 412000, {"executed_fields_only": [False, True]}, 1.0),
                                      (f1, 420000, {"eye_row_convention": ["legacy", "physical"]}, 2.0),
                                      (back, 430000, {"eye_row_convention": ["physical", "legacy"]}, 3.0)):
        folder.mkdir(parents=True)
        (folder / "replay_training_stats.jsonl").write_text(
            json.dumps({"type": "motor_interface_change", "step": step, "timestamp": ts,
                        "changes": changes}) + "\n" + json.dumps(rec(step + 100)) + "\n", encoding="utf-8")
    monkeypatch.setattr(MB, "EXP", exp)
    assert MB.resolve_after_step("auto", {41})[0] == 420000, "a revert is not a deploy's N"
    assert MB.resolve_after_step("auto:executed_fields_only", {41})[0] == 412000
    step, used = MB.resolve_after_step("auto:eye_row_convention", {41})
    assert step == 420000 and used["folder"] == f1.name
    assert MB.resolve_after_step("1234", {41}) == (1234, None)
    with pytest.raises(SystemExit):
        MB.resolve_after_step("auto", {40})
    assert [r["step"] for _, r in MB.stats_rows({41}, 412000)] == [412100, 420100, 430100]


# ------------------------------------------------------------------ the eye-stop line, per block
V_RANGE_DEG = (-47.0, 33.0)
H_RANGE_DEG = (-45.0, 45.0)


def write_eye_chunks(run, steps, vertical_deg, horizontal_deg=5.0, bodies=8, per_chunk=850):
    """Minimal observer chunks (observer_record's format: step, header name, qpos delta-coded) with the
    left eye's two joints at qpos 0 and 1."""
    from mimo_bridge.observer_record import encode_series
    quantum = np.array([1e-6, 1e-6])
    steps = np.asarray(steps)
    vertical_deg = np.asarray(vertical_deg, dtype=np.float64)
    for b in range(bodies):
        folder = Path(run) / "observer" / "world" / ("b%d" % b)
        folder.mkdir(parents=True, exist_ok=True)
        header = {"joints": [
            {"name": MB.EYE_H, "qpos": 0, "range": [math.radians(x) for x in H_RANGE_DEG]},
            {"name": MB.EYE_V, "qpos": 1, "range": [math.radians(x) for x in V_RANGE_DEG]}],
            "qpos_quantum": quantum.tolist(), "nmocap": 0, "objects": []}
        (folder / "header_test.json").write_text(json.dumps(header), encoding="utf-8")
        for i in range(0, len(steps), per_chunk):
            s = steps[i:i + per_chunk].astype(np.int64)
            v = np.radians(vertical_deg[i:i + per_chunk])
            qpos = np.stack([np.full(len(s), math.radians(horizontal_deg)), v], axis=1)
            np.savez(folder / ("chunk_%09d_1_%05d.npz" % (s[0], i // per_chunk)), step=s,
                     header=np.asarray("header_test.json"), **encode_series(qpos, quantum, "qpos"))


def test_the_eye_stop_line_is_judged_per_block_where_a_folder_share_would_miss_it(tmp_path):
    """The first branch's finding 4: 20 blocks, the last 2 with the vertical eye angle within 2 degrees
    of a stop in 50 of every 100 records. Over the folder: 5.0 of 100; per block: 50."""
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 2), switch=N)
    steps = np.arange(N, N + 20 * 1700 + 100)          # one chunk past the last block's end
    vert = np.full(len(steps), 5.0)
    late = (steps >= N + 18 * 1700) & (steps < N + 20 * 1700)
    vert[late & (steps % 2 == 0)] = 32.0                    # 1 degree from the +33 stop
    write_eye_chunks(run, steps, vert)
    S, H, V, rh, rv = MB.eye_series(run, N)
    k = S < N + 20 * 1700
    assert abs(100.0 * MB.near_stop(V[k], rv).mean() - 5.0) < 1e-9
    # the line's own arithmetic over 20 blocks: a watch without an end (the deploy's watch ends after 6,000
    # updates, test_the_watch_ends_after_its_6000_updates)
    why, state = W.watch_once(run, saved_for(tmp_path, run), watch_steps=0)
    assert why and "eye stop" in why and "block of steps %d" % (N + 18 * 1700) in why
    assert (run / "control.json").exists()
    rows = state["eye_blocks"]
    assert [b["vertical_near_stop_per_100"] for b in rows[:18]] == [0.0] * 18
    assert rows[18]["vertical_near_stop_per_100"] == 50.0
    # the crossing block is recorded as judged once its stop is written, keyed with its run
    assert W.block_key(N + 18 * 1700, ["deploy"]) in W.load_state(run)["judged_blocks"]


def test_a_block_split_across_a_stop_is_judged_over_the_joined_records(tmp_path):
    """The deploy branch's finding 1, its failing input: the vertical eye angle runs to a stop 900 steps
    into a block and stays there; the scheduler's window stops the run 700 steps later; the
    continuation holds the block's last 100 steps. Per folder, neither holds a judgeable block."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 16), switch=N)
    a = np.arange(N, N + 1600)
    va = np.where(a >= N + 900, 32.5, 5.0)
    write_eye_chunks(deploy, a, va, per_chunk=100)
    cont = make_run(exp, "deploy_resume_1", windows(N + 1700, 3), parent=deploy, checkpoint=ckpt(N + 1600))
    b = np.arange(N + 1600, N + 1900)                   # the block's last 100 steps and one chunk beyond
    write_eye_chunks(cont, b, np.full(len(b), 32.5), per_chunk=100)
    # judged per folder, as the first build did: the deploy run's block never completes, and the
    # continuation's piece of it is under 2,000 records
    assert not MB.eye_blocks(*MB.eye_series(deploy, N), N)[0]["complete"]
    assert MB.eye_blocks(*MB.eye_series(cont, N), N)[0].get("vertical_near_stop_per_100") is None
    why, state = W.watch_once(deploy, saved_for(tmp_path, cont))
    assert why and "eye stop" in why and "block of steps %d to %d" % (N, N + 1699) in why
    first = state["eye_blocks"][0]
    assert first["records"] == 8 * 1700
    assert abs(first["vertical_near_stop_per_100"] - 100.0 * 800 / 1700) < 0.01
    assert (cont / "control.json").exists()


def test_an_incomplete_block_waits_for_the_chunk_after_its_end(tmp_path):
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 2), switch=N)
    steps = np.arange(N, N + 1700 + 900)
    vert = np.full(len(steps), 5.0)
    vert[steps >= N + 1700] = -46.5                          # the second block: every record at a stop
    write_eye_chunks(run, steps, vert)
    memory = W.Memory()
    why, state = W.watch_once(run, saved_for(tmp_path, run), memory=memory)
    assert why is None, "the second block holds 900 of its 1,700 steps: not judged yet"
    assert memory.judged_blocks == {W.block_key(N, ["deploy"])}
    assert state["eye_blocks"][1]["complete"] is False
    # records exactly to the block's last step: still waiting for the chunk after it
    more = np.arange(N + 2600, N + 3400)
    write_eye_chunks(run, more, np.full(len(more), -46.5), per_chunk=800)
    why, _ = W.watch_once(run, saved_for(tmp_path, run), memory=memory)
    assert why is None
    beyond = np.arange(N + 3400, N + 3500)
    write_eye_chunks(run, beyond, np.full(len(beyond), 5.0), per_chunk=100)
    why, _ = W.watch_once(run, saved_for(tmp_path, run), memory=memory)
    assert why and "block of steps %d" % (N + 1700) in why


def test_the_watch_ends_after_its_6000_updates(tmp_path, capsys):
    """The operator's decision of 2026-09-25 (F4 and F1 together, one deploy stop, one watch): the watch is
    the 6,000 updates after the switch. A crossing after them is outside the watch; once every window and
    every eye block inside it is judged, the phase is 'complete', the watcher ends and --ensure starts none."""
    assert W.WATCH_STEPS == 6000
    exp = tmp_path / "exp"
    # windows to N + 5,000, safe: the watch waits for the rest
    run = make_run(exp, "deploy", windows(N + 100, 50), switch=N)
    steps = np.arange(N, N + 5000)
    write_eye_chunks(run, steps, np.full(len(steps), 5.0), per_chunk=500)
    saved = saved_for(tmp_path, run)
    why, state = W.watch_once(run, saved)
    assert why is None and state["phase"] == "watching" and state["watch_end_step"] == N + 6000
    assert "stats windows up to step %d" % (N + 6000) in state["watch_awaits"]
    assert "from step %d" % (N + 3400) in state["watch_awaits"] or str(N + 3400) in state["watch_awaits"]
    W.print_state(state)
    assert "it awaits" in capsys.readouterr().out
    # the windows to N + 6,000, safe; three crossing windows AFTER it are outside the watch
    append(run, windows(N + 5100, 10) + windows(N + 6100, 3, **BAD))
    # the eye blocks inside the watch: the last starts at N + 5,100 and ends at N + 6,799, judged once the
    # records pass that end by a chunk
    more = np.arange(N + 5000, N + 6900)
    write_eye_chunks(run, more, np.full(len(more), 5.0), per_chunk=950)
    why, state = W.watch_once(run, saved)
    assert why is None, "the crossing after the watch's end is not judged"
    assert not (run / "control.json").exists()
    assert state["phase"] == "complete", state.get("watch_awaits")
    assert "no kill line was crossed in the 6000 updates after the switch" in state["status"]
    assert [b["first_step"] for b in state["eye_blocks"]] == [N, N + 1700, N + 3400, N + 5100]
    assert W.load_state(run)["complete"]["steps"] == [N, N + 6000]
    # --ensure starts no watcher for a complete watch; main ends with 6
    started = []
    got, msg = W.ensure([], run, 60.0, held=lambda r: False, start=lambda a, r: started.append(1) or 1)
    assert got is False and not started and "the watch is complete" in msg
    assert W.main(["--deploy-run", str(run), "--saved", str(saved), "--once"]) == 6
    # the dashboards say the watch finished, and never call its ended watcher silent
    from mimo_bridge.tools.lineage_series import motor_watch_note
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    note = motor_watch_note(saved, now=W.time.time() + 3600)
    assert note.startswith("The motor fixes kill-line watcher finished its watch of the 6,000 updates after the "
                           "switch (lineage steps %s to %s) with no kill line crossed"
                           % (format(N, ","), format(N + 6000, ",")))
    assert "last polled" not in note and not bare_decimals(note)
    # without an end (--watch-steps 0) the same records are a stop
    run2 = make_run(exp, "deploy_no_end", windows(N + 100, 60) + windows(N + 6100, 3, **BAD), switch=N)
    why, _ = W.watch_once(run2, saved_for(tmp_path, run2), watch_steps=0)
    assert why and "body head" in why


def test_a_block_left_without_its_observer_records_is_settled_blind(tmp_path):
    """The observer records stop early; training runs a whole block past the last block's end: the watch
    completes and names the blind blocks, instead of waiting for records that will not come."""
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 60) + windows(N + 6100, 40), switch=N)
    steps = np.arange(N, N + 3500)                      # the first two blocks and a chunk past the second
    write_eye_chunks(run, steps, np.full(len(steps), 5.0), per_chunk=500)
    saved = saved_for(tmp_path, run)
    why, state = W.watch_once(run, saved)
    assert why is None and state["phase"] == "complete"
    assert state["complete"]["eye_blind_blocks"] == [N + 3400, N + 5100]
    assert "blind for the blocks from step %d, %d" % (N + 3400, N + 5100) in state["status"]
    from mimo_bridge.tools.lineage_series import motor_watch_note
    assert ("could not judge the eye blocks that start at lineage steps %s, %s"
            % (format(N + 3400, ","), format(N + 5100, ","))) in motor_watch_note(saved)


def test_the_line_says_when_it_is_blind(tmp_path, capsys):
    """Review finding 9: a block with too few records, and a stretch with no observer record at all."""
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 40), switch=N)
    steps = np.arange(N, N + 1900)
    write_eye_chunks(run, steps, np.full(len(steps), 5.0), bodies=1, per_chunk=100)   # 1,900 records
    why, state = W.watch_once(run, saved_for(tmp_path, run))
    assert why is None
    assert any("under the 2000 a block needs" in s for s in state["eye_blind"])
    assert any("while training reached step %d" % (N + 4000) in s for s in state["eye_blind"])
    W.print_state(state)
    assert "EYE-STOP LINE BLIND" in capsys.readouterr().out
    run2 = make_run(exp, "deploy2", windows(N + 100, 40), switch=N)
    _, state = W.watch_once(run2, saved_for(tmp_path, run2))
    assert any("no observer record since step %d" % N in s for s in state["eye_blind"])


def test_the_pre_deploy_baselines_are_copied_for_the_hourly_watcher(tmp_path):
    exp = tmp_path / "exp"
    pre = exp / "pre_deploy"
    pre.mkdir(parents=True)
    (pre / "watcher_baselines.json").write_text(json.dumps(
        {"vision": 0.01, "tactile": 0.04, "measured_at_step": 400000}), encoding="utf-8")
    deploy = make_run(exp, "deploy", windows(N + 100, 1), switch=N)
    early = make_run(exp, "deploy_resume_1", windows(N + 2100, 1), parent=deploy, checkpoint=ckpt(N + 2000))
    late = make_run(exp, "deploy_resume_2", windows(N + 7100, 1), parent=early, checkpoint=ckpt(N + 7000))
    own = {"vision": 0.5, "measured_at_step": N + 900}
    (early / "watcher_baselines.json").write_text(json.dumps(own), encoding="utf-8")
    _, state = W.watch_once(deploy, saved_for(tmp_path, late), baseline_run=pre)
    got = json.loads((deploy / "watcher_baselines.json").read_text(encoding="utf-8"))
    assert got["vision"] == 0.01 and got["measured_at_step"] == 400000 and got["copied_from"]
    assert json.loads((early / "watcher_baselines.json").read_text(encoding="utf-8")) == own, \
        "a baseline the hourly watcher measured itself is never replaced"
    assert not (late / "watcher_baselines.json").exists(), "past the replay ring's turnover"
    assert "past the replay ring's turnover" in state["baselines"]["deploy_resume_2"]
    # the hourly watcher reads the copy as its own baselines
    spec = importlib.util.spec_from_file_location(
        "hourly_watch_copy", DOCS.parent / "h84_launch_2026-09-18" / "hourly_watch.py")
    hw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hw)
    hw.BASE_PATH = str(deploy / "watcher_baselines.json")
    assert hw.load_baselines()["vision"] == 0.01


def test_the_printed_lines_carry_units_or_shares_of_a_line(tmp_path, capsys):
    run = make_run(tmp_path / "exp", "deploy", windows(N + 100, 3), switch=N)
    _, state = W.watch_once(run, saved_for(tmp_path, run))
    W.print_state(state)
    out = capsys.readouterr().out
    assert "cm under the error with another body's order" in out
    assert "% of its kill line" in out
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    assert not bare_decimals(out), bare_decimals(out)


def test_a_deploy_relaunch_that_also_changed_the_decoder_rows_or_the_limb_flags_is_named(tmp_path, capsys):
    """The deploy branch's review, finding 5: --position-rows and the limb flags switch in a relaunch
    of their own. The watcher compares the deploy run's args.json with the run it resumed from and
    says NOT COVERED; it writes no stop for it."""
    exp = tmp_path / "exp"
    pre = exp / "pre_deploy"
    (pre / "checkpoints" / "ckpt_step_410000").mkdir(parents=True)
    (pre / "args.json").write_text(json.dumps({"position_rows": "all", "report_every": 100}), encoding="utf-8")
    run = make_run(exp, "deploy", windows(N + 100, 3), switch=N)
    resume = str(pre / "checkpoints" / "ckpt_step_410000")
    (run / "args.json").write_text(json.dumps({"resume": resume, "position_rows": "all",
                                               "executed_fields_only": "on"}), encoding="utf-8")
    assert W.same_relaunch_changes(run) == {}
    (run / "args.json").write_text(json.dumps({"resume": resume, "position_rows": "no_head_pelvis",
                                               "reach_projection": "arms"}), encoding="utf-8")
    assert W.same_relaunch_changes(run) == {"position_rows": ["all", "no_head_pelvis"],
                                            "reach_projection": ["off", "arms"]}
    why, state = W.watch_once(run, saved_for(tmp_path, run))
    assert why is None and not (run / "control.json").exists()
    W.print_state(state)
    out = capsys.readouterr().out
    assert "NOT COVERED" in out and "--position-rows from all to no_head_pelvis" in out
    # without a readable predecessor it says so; --baseline-run stands in for it
    (run / "args.json").write_text(json.dumps({"resume": str(tmp_path / "gone" / "checkpoints" / "ckpt_step_1")}),
                                   encoding="utf-8")
    assert W.same_relaunch_changes(run).startswith("unchecked")
    assert W.same_relaunch_changes(run, baseline_run=pre) == {}


# ------------------------------------------------------------------ the second review's findings 2, 3, 6, 7
class Trainers:
    """A stand-in for the process list: {run name: [process ids]}; the watcher asks per run."""

    def __init__(self, **runs):
        self.runs = dict(runs)

    def __call__(self, run):
        return list(self.runs.get(Path(run).name, []))


def consume(run, cmd_owner=None):
    """What the trainer does when it reads control.json: renames it to control_done_<n>.json."""
    src = Path(run) / "control.json"
    n = len(list(Path(run).glob("control_done_*.json")))
    src.replace(Path(run) / ("control_done_%d.json" % (1_790_000_000 + n)))


def test_a_written_stop_is_confirmed_only_when_a_trainer_read_it_and_none_remains(tmp_path, monkeypatch):
    """The second review, finding 2, its failing input: the active run changes between the watcher's
    read and its write (the supervisor relaunched a continuation), so control.json lands where no
    trainer reads it. The watch goes on polling and writes into the run the trainer runs in."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    # the continuation's launch began before the stop is written (its receipt carries the supervisor's time)
    cont = make_run(exp, "deploy_resume_1", [], parent=deploy, checkpoint=ckpt(N + 400))
    receipt = json.loads((cont / "continuation_receipt.json").read_text(encoding="utf-8"))
    receipt["timestamp"] = W.time.time() - 5
    (cont / "continuation_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    trainers = Trainers(deploy_resume_1=[99])
    monkeypatch.setattr(W, "trainers_of", trainers)
    memory = W.Memory()
    why, state = W.watch_once(deploy, saved, memory=memory)          # read: deploy active; written there
    assert why and state["phase"] == "confirming" and (deploy / "control.json").exists()
    sid = json.loads((deploy / "control.json").read_text(encoding="utf-8"))["stop_id"]
    sup = json.loads((saved / "supervisor_control.json").read_text(encoding="utf-8"))
    assert sup["cmd"] == "pause" and sup["stop_id"] == sid and sup["writer"] == W.WRITER
    # ... while the supervisor had already launched the continuation and made it active
    saved_for(tmp_path, cont, control=sup)
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert why is None and state["phase"] == "confirming"
    assert json.loads((cont / "control.json").read_text(encoding="utf-8"))["stop_id"] == sid
    assert str(cont) in state["open_stop"]["written_to"]
    assert W.waiting_lines(state)[0].startswith("STOP WRITTEN") and "deploy_resume_1" in W.waiting_lines(state)[0]
    # the trainer reads it but still runs (saving its snapshot): not yet confirmed, nothing rewritten
    consume(cont)
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert state["phase"] == "confirming" and not (cont / "control.json").exists()
    # it has ended: confirmed; the watch would end here
    trainers.runs = {}
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert state["phase"] == "confirmed" and W.load_state(deploy)["stops"][0]["confirmed"]
    assert W.Memory(W.load_state(deploy)).open_stop() is None


def test_the_schedulers_snapshot_request_over_the_stop_is_followed_by_the_stop_again(tmp_path, monkeypatch):
    """The scheduler writes {"cmd": "snapshot"} with a replace, not an exclusive create: when it lands
    over the watcher's control.json the trainer reads the snapshot request and trains on. The watcher
    writes its stop again once that request is read, and never over it."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    (deploy / "args.json").write_text("{}", encoding="utf-8")
    os.utime(deploy / "args.json", (W.time.time() - 60, W.time.time() - 60))   # launched before the stop
    saved = saved_for(tmp_path, deploy)
    trainers = Trainers(deploy=[11])
    monkeypatch.setattr(W, "trainers_of", trainers)
    memory = W.Memory()
    why, _ = W.watch_once(deploy, saved, memory=memory)
    assert why
    (deploy / "control.json").write_text(json.dumps({"cmd": "snapshot"}), encoding="utf-8")   # the scheduler
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert json.loads((deploy / "control.json").read_text(encoding="utf-8")) == {"cmd": "snapshot"}, \
        "another program's control.json is never replaced"
    consume(deploy)                                   # the trainer reads the snapshot request, trains on
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert json.loads((deploy / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    consume(deploy)
    trainers.runs = {}
    _, state = W.watch_once(deploy, saved, memory=memory)
    assert state["phase"] == "confirmed"


def test_a_run_launched_after_the_stop_is_not_chased_and_a_lifted_pause_releases_it(tmp_path, monkeypatch):
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    trainers = Trainers(deploy=[11])
    monkeypatch.setattr(W, "trainers_of", trainers)
    memory = W.Memory()
    assert W.watch_once(deploy, saved, memory=memory)[0]
    # the trainer died before reading the stop; the operator relaunches by hand AFTER the stop (its args
    # are written at its start), under the stop's pause
    trainers.runs = {"hand": [12]}
    hand = make_run(exp, "hand", windows(N + 500, 1))
    (hand / "args.json").write_text(json.dumps({"resume": str(deploy / "checkpoints" / ("ckpt_step_%d" % (N + 400)))}),
                                    encoding="utf-8")
    _, state = W.watch_once(deploy, saved, memory=memory)
    assert state["phase"] == "confirming" and not (hand / "control.json").exists(), \
        "a run launched after the stop is someone else's decision"
    # the operator lifts the stop's pause: the stop is released; the watch goes on (the third review,
    # finding A) and judges the relaunch from its first full windows
    sup = json.loads((saved / "supervisor_control.json").read_text(encoding="utf-8"))
    (saved / "supervisor_control.json").write_text(json.dumps(dict(sup, cmd="run")), encoding="utf-8")
    _, state = W.watch_once(deploy, saved, memory=memory)
    assert state["phase"] == "released" and W.load_state(deploy)["stops"][0]["released"]
    append(hand, windows(N + 600, 3, **BAD))
    saved_for(tmp_path, hand, control=dict(sup, cmd="run"))
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert why and "hand" in why and (hand / "control.json").exists()


def test_a_stop_that_landed_in_a_stopped_run_is_confirmed_and_the_watch_goes_on(tmp_path, monkeypatch):
    """The third review, finding A, its scenario A (probe_watch_rev3.py): a stop that waited for the
    scheduler's saved window is written at the watcher's first poll after the window releases, before the
    supervisor's relaunch (37 to 53 s after, measured), into the stopped run; its pause keeps the supervisor
    from relaunching, and no trainer ever reads it. Before: 'confirming' at every poll, and the relaunch
    by hand that continues the deploy ran unwatched (the watch ended at the release). Now: confirmed after
    two polls with no trainer launched before the stop, the last at least QUIET_SECONDS after the write;
    the watch goes on and stops the relaunch on its own crossing."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    trainers = Trainers()                      # the deploy trainer already stopped for the window
    monkeypatch.setattr(W, "trainers_of", trainers)
    saved = saved_for(tmp_path, deploy, scheduler={"comparison_window": {"state": "ready"}},
                      control={"cmd": "pause", "ability_window_token": "t", "reason": "Saved comparison window"})
    memory = W.Memory()
    why, st = W.watch_once(deploy, saved, memory=memory)
    assert why is None and st["phase"] == "waiting"
    # the window releases (the scheduler restores {"cmd": "run"}); the watcher polls before the relaunch
    saved_for(tmp_path, deploy, scheduler={"comparison_window": {"state": "released"}}, control={"cmd": "run"})
    why, st = W.watch_once(deploy, saved, memory=memory)
    assert why and st["open_stop"]["written_to"] == [str(deploy)]
    # two quiet polls within QUIET_SECONDS of the write: not yet (a relaunch could still be in flight)
    for _ in range(2):
        _, st = W.watch_once(deploy, saved, memory=memory)
    assert st["phase"] == "confirming" and memory.open_stop()["quiet_polls"] == 2
    # a trainer launched before the stop running on one poll starts the count again
    trainers.runs = {"deploy": [5]}
    _, st = W.watch_once(deploy, saved, memory=memory)
    assert st["phase"] == "confirming" and memory.open_stop()["quiet_polls"] == 0
    trainers.runs = {}
    # the write a minute back (in the state file, which every poll reads back), and two more quiet polls:
    # confirmed, as landed in a stopped run
    kept = W.load_state(deploy)
    kept["stops"][0]["time"] -= W.QUIET_SECONDS
    (deploy / W.STATE_NAME).write_text(json.dumps(kept), encoding="utf-8")
    memory.open_stop()["time"] = kept["stops"][0]["time"]
    _, st = W.watch_once(deploy, saved, memory=memory)
    assert st["phase"] == "confirming"
    _, st = W.watch_once(deploy, saved, memory=memory)
    assert st["phase"] == "confirmed"
    stop = W.load_state(deploy)["stops"][0]
    assert stop["confirmed_how"].startswith("landed in a stopped run") and "stop_reason.json" in stop["confirmed_how"]
    assert W.Memory(W.load_state(deploy)).open_stop() is None
    # the procedure's step 8: diagnosis finds no harm; a relaunch by hand from the stop's checkpoint (launched
    # after the stop), the supervisor reattached and the pause lifted; the same fault recurs there
    hand = make_run(exp, "deploy_by_hand", windows(N + 500, 1) + windows(N + 600, 3, **BAD))
    (hand / "args.json").write_text(json.dumps({"resume": str(deploy / "checkpoints" / ("ckpt_step_%d" % (N + 400)))}),
                                    encoding="utf-8")
    trainers.runs = {"deploy_by_hand": [77]}
    sup = json.loads((saved / "supervisor_control.json").read_text(encoding="utf-8"))
    saved_for(tmp_path, hand, control=dict(sup, cmd="run"))
    why, st = W.watch_once(deploy, saved, memory=memory)
    assert why and "deploy_by_hand" in why and (hand / "control.json").exists()
    assert st["closed_runs"] == {"deploy": N + 400}
    # --ensure names the new stop, written and not yet read
    W._write_state(deploy, dict(W.load_state(deploy), pid=7, time=W.time.time()))
    msg = W.ensure([], deploy, 60.0, held=lambda r: True, start=lambda a, d: 0)[1]
    assert msg.startswith("STOP WRITTEN") and "deploy_by_hand" in msg.splitlines()[0]
    # each stop's crossing run is closed: the windows a waiting stop let it write are never a new crossing
    # that could stop a continuation
    assert W.Memory(W.load_state(deploy)).closed_runs() == {"deploy": N + 400, "deploy_by_hand": N + 800}


def test_a_plain_pause_while_the_watched_trainer_runs_does_not_hold_the_stop(tmp_path, monkeypatch):
    """The third review, finding E: the deploy trainer runs under the coordinator's plain pause from the
    procedure's step 4 until step 6 lifts it, and a stop waited for that pause with no end. A plain pause
    (no scheduler window, no unconsumed control.json) while the watched trainer runs is no hold now:
    control.json and stop_reason.json are written, the pause is kept as it is, and lifting it later does
    not 'release' the stop."""
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    pause = {"cmd": "pause", "reason": "the deploy's own steps 4 to 6"}
    trainers = Trainers(deploy=[5])
    monkeypatch.setattr(W, "trainers_of", trainers)
    saved = saved_for(tmp_path, deploy, control=pause)
    memory = W.Memory()
    why, st = W.watch_once(deploy, saved, memory=memory)
    assert why and json.loads((deploy / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    assert json.loads((deploy / "stop_reason.json").read_text(encoding="utf-8"))["resume_policy"] == "repair_required"
    assert json.loads((saved / "supervisor_control.json").read_text(encoding="utf-8")) == pause, "kept as it is"
    stop = W.load_state(deploy)["stops"][0]
    assert stop["pause_kept"] == pause["reason"] and stop["pause_written"] is False
    # the coordinator lifts the pause (step 6) before the trainer read the stop: not 'released'
    (saved / "supervisor_control.json").write_text(json.dumps({"cmd": "run"}), encoding="utf-8")
    _, st = W.watch_once(deploy, saved, memory=memory)
    assert st["phase"] == "confirming"
    consume(deploy)
    trainers.runs = {}
    _, st = W.watch_once(deploy, saved, memory=memory)
    assert st["phase"] == "confirmed"
    from mimo_bridge.tools.training_continuation import pause_reason
    assert pause_reason(tmp_path / "no_saved_here", deploy) is not None, "the supervisor does not relaunch it"
    # the same pause with no trainer of the run running still holds the stop, and so do the scheduler's
    # window and a coordinated pause's own control.json while a trainer runs
    for name, scheduler, own_control, running in (
            ("idle", None, False, []), ("window", {"comparison_window": {"state": "saving"}}, False, [5]),
            ("coordinated", None, True, [5])):
        run = make_run(exp, "deploy_" + name, windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
        if own_control:
            (run / "control.json").write_text(json.dumps({"cmd": "snapshot_and_stop"}), encoding="utf-8")
        monkeypatch.setattr(W, "trainers_of", Trainers(**{"deploy_" + name: running}))
        (tmp_path / name).mkdir()
        s2 = saved_for(tmp_path / name, run, scheduler=scheduler, control=pause)
        why, st = W.watch_once(run, s2, memory=W.Memory())
        assert why is None and st["phase"] == "waiting", name
        assert json.loads((s2 / "supervisor_control.json").read_text(encoding="utf-8")) == pause


def test_a_revert_is_not_followed_and_end_ends_the_watch(tmp_path, monkeypatch):
    """The third review, point G: the watcher followed any relaunch from a watched run's checkpoint, a
    revert too, and never ended on its own without a stop. F4 and F1 now deploy together, so one watcher
    watches the combined form: a relaunch that reverts a deploy setting is not followed (and is named),
    nor anything continuing it; --end ends the watch and --ensure starts none after it."""
    monkeypatch.setattr(W, "trainers_of", Trainers())
    exp = tmp_path / "exp"
    on = {"eye_row_convention": "physical", "executed_fields_only": "on"}
    deploy = make_run(exp, "deploy", windows(N + 100, 3), switch=N)
    (deploy / "args.json").write_text(json.dumps(on), encoding="utf-8")
    cont = make_run(exp, "deploy_resume_1", windows(N + 400, 2), parent=deploy, checkpoint=ckpt(N + 300))
    (cont / "args.json").write_text(json.dumps(on), encoding="utf-8")
    revert = make_run(exp, "deploy_revert", windows(N + 700, 4, **BAD))
    (revert / "args.json").write_text(json.dumps({"resume": str(cont / "checkpoints" / ("ckpt_step_%d" % (N + 600))),
                                                  "eye_row_convention": "legacy", "executed_fields_only": "off"}),
                                      encoding="utf-8")
    make_run(exp, "deploy_revert_resume_1", windows(N + 1200, 4, **BAD), parent=revert, checkpoint=ckpt(N + 1100))
    # a launch whose trainer has not written its args.json yet cannot be told from the deploy's: followed
    early = make_run(exp, "deploy_resume_2", [], parent=cont, checkpoint=ckpt(N + 600))
    not_followed = {}
    assert [p.name for p in W.watched_runs(deploy, not_followed)] == ["deploy", "deploy_resume_1", "deploy_resume_2"]
    assert not_followed == {"deploy_revert": {"eye_row_convention": ["physical", "legacy"],
                                              "executed_fields_only": ["on", "off"]}}
    saved = saved_for(tmp_path, early)
    why, st = W.watch_once(deploy, saved, memory=W.Memory())
    assert why is None and "deploy_revert" in st["not_followed"] and not (early / "control.json").exists()
    # --end: the watch ends at its next poll, and --ensure starts none after it
    assert W.main(["--deploy-run", str(deploy), "--saved", str(saved), "--end", "reverted to the checkpoint before"]) == 0
    why, st = W.watch_once(deploy, saved, memory=W.Memory())
    assert st["phase"] == "ended" and "reverted to the checkpoint before" in st["status"]
    assert W.main(["--deploy-run", str(deploy), "--saved", str(saved), "--once"]) == 7
    got, msg = W.ensure([], deploy, 60.0, held=lambda r: False, start=lambda a, d: 1)
    assert not got and "the watch was ended" in msg
    from mimo_bridge.tools.lineage_series import motor_watch_note
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    note = motor_watch_note(saved, now=W.time.time() + 3600)
    assert "was ended by hand" in note and "does not watch deploy_revert" in note and "last polled" not in note
    assert not bare_decimals(note)


def test_the_watch_goes_on_after_a_confirmed_stop_and_completes(tmp_path, monkeypatch):
    """A confirmed stop no longer ends the watch: a continuation that runs the rest of the 6,000 updates
    without a crossing completes it, and the completion names the stop."""
    monkeypatch.setattr(W, "trainers_of", Trainers())
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    memory = W.Memory()
    assert W.watch_once(deploy, saved, memory=memory)[0]
    consume(deploy)
    assert W.watch_once(deploy, saved, memory=memory)[1]["phase"] == "confirmed"
    hand = make_run(exp, "deploy_by_hand", windows(N + 500, 56))
    (hand / "args.json").write_text(json.dumps({"resume": str(deploy / "checkpoints" / ("ckpt_step_%d" % (N + 400)))}),
                                    encoding="utf-8")
    steps = np.arange(N, N + 6900)
    write_eye_chunks(hand, steps, np.full(len(steps), 5.0), per_chunk=950)
    saved_for(tmp_path, hand, control={"cmd": "run"})
    why, st = W.watch_once(deploy, saved, memory=memory)
    assert why is None and st["phase"] == "complete", st.get("watch_awaits")
    assert "1 stop was written" in st["status"] and len(st["complete"]["stops"]) == 1
    from mimo_bridge.tools.lineage_series import motor_watch_note
    note = motor_watch_note(saved, now=W.time.time() + 3600)
    assert "after 1 stop it wrote" in note and "last polled" not in note


def test_two_watchers_never_erase_a_written_stop_nor_stop_a_continuation_on_it(tmp_path, monkeypatch):
    """The second review, finding 3, reproduced with its probe (probe_two_watchers.py): A writes the
    stop; B, started before A's first poll, rewrote the state file from its own memory (0 written stops)
    and later stopped the continuation on the old crossing. The state file is now merged at every write
    and read back at every poll (and main() holds a lock: one watcher per deploy run)."""
    monkeypatch.setattr(W, "trainers_of", Trainers())
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    A, B = W.Memory(W.load_state(deploy)), W.Memory(W.load_state(deploy))
    assert W.watch_once(deploy, saved, memory=A)[0]
    consume(deploy)                                   # the trainer read it and stopped
    _, st = W.watch_once(deploy, saved, memory=B)
    assert st["phase"] == "confirmed" and len(W.load_state(deploy)["stops"]) == 1
    # a stale write (a memory with no stop) cannot erase it either
    W._write_state(deploy, {"stops": [], "judged_blocks": []})
    assert len(W.load_state(deploy)["stops"]) == 1
    cont = make_run(exp, "deploy_by_hand", windows(N + 500, 1))
    (cont / "args.json").write_text(json.dumps({"resume": str(deploy / "checkpoints" / ("ckpt_step_%d" % (N + 420)))}),
                                    encoding="utf-8")
    saved_for(tmp_path, cont, control={"cmd": "run"})
    why, _ = W.watch_once(deploy, saved, memory=B)
    assert why is None and not (cont / "control.json").exists()
    assert W.Memory(W.load_state(deploy)).judged_until() == {"deploy": N + 400}


def test_only_the_stopped_runs_windows_are_skipped_after_a_stop(tmp_path, monkeypatch):
    """The second review, point 7: a continuation from an OLDER checkpoint (the hourly one) re-runs the
    steps before the old stop; its windows there are judged, and three of them past a line stop it."""
    monkeypatch.setattr(W, "trainers_of", Trainers())
    exp = tmp_path / "exp"
    deploy = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 5, **BAD), switch=N)
    saved = saved_for(tmp_path, deploy)
    memory = W.Memory()
    assert W.watch_once(deploy, saved, memory=memory)[0]
    consume(deploy)
    assert W.watch_once(deploy, saved, memory=memory)[1]["phase"] == "confirmed"
    # the operator continues by hand from the checkpoint at N + 100, and the same fault recurs
    cont = make_run(exp, "deploy_from_older", windows(N + 200, 1) + windows(N + 300, 3, **BAD))
    (cont / "args.json").write_text(json.dumps({"resume": str(deploy / "checkpoints" / ("ckpt_step_%d" % (N + 100)))}),
                                    encoding="utf-8")
    saved_for(tmp_path, cont, control={"cmd": "run"})
    why, state = W.watch_once(deploy, saved, memory=memory)
    assert why and "deploy_from_older" in why and "up to step %d" % (N + 500) in why
    assert state["judged_until"] == {"deploy": N + 400}


def test_an_eye_block_rerun_by_an_older_continuation_is_judged_again(tmp_path):
    """A judged block is keyed by its first step AND the runs whose records it held: a continuation from
    an older checkpoint replaces the records, so the block is judged again."""
    run_a = ["deploy"]
    key_a = W.block_key(N, run_a)
    assert key_a == "%d@deploy" % N and W.block_key(N, ["deploy", "deploy_from_older"]) != key_a
    S = np.arange(N, N + 1800)
    series = (S, np.full(len(S), 5.0), np.full(len(S), 5.0), H_RANGE_DEG, V_RANGE_DEG,
              np.full(len(S), "deploy_from_older", dtype=object))
    judged = {key_a}
    why, rows, _ = W.judge_eye_blocks(series, N, judged, min_records=100)
    assert rows[0]["key"] == W.block_key(N, ["deploy_from_older"]) and rows[0]["key"] in judged


def test_holds_and_the_stops_writes_follow_what_the_supervisor_reads(tmp_path):
    """The second review, findings 6 and point 7: a task the scheduler runs outside a saved window is no
    hold; an unreadable supervisor_control.json, or one without a cmd, is a pause (as the supervisor
    reads it); the stop's pause keeps every other key, and leaves an unreadable file as it is."""
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, run, scheduler={"comparison_window": {"state": "released"},
                                                "active_task": "hand_following"})
    assert W.write_hold(saved, run) is None
    (saved / "supervisor_control.json").write_bytes("{\"cmd\": \"run\"}".encode("utf-16"))
    assert "unreadable" in W.write_hold(saved, run)
    (saved / "supervisor_control.json").write_text(json.dumps({"archiver": "off"}), encoding="utf-8")
    assert "no cmd" in W.write_hold(saved, run)
    # the pause keeps the archiver switch
    (saved / "supervisor_control.json").write_text(json.dumps({"cmd": "run", "archiver": "off"}), encoding="utf-8")
    stop = W.stop_run(run, saved, "motor fixes kill line crossed: test")
    sup = json.loads((saved / "supervisor_control.json").read_text(encoding="utf-8"))
    assert sup["cmd"] == "pause" and sup["archiver"] == "off" and stop["pause_written"]
    # an unreadable file is left as it is; stop_reason.json still keeps the supervisor away
    run2 = make_run(exp, "deploy2", [], switch=N)
    raw = "{\"cmd\": \"run\", \"archiver\": \"off\"}".encode("utf-16")
    (saved / "supervisor_control.json").write_bytes(raw)
    stop = W.stop_run(run2, saved, "motor fixes kill line crossed: test")
    assert (saved / "supervisor_control.json").read_bytes() == raw and not stop["pause_written"]
    assert any("unreadable" in n for n in stop["notes"])
    assert json.loads((run2 / "stop_reason.json").read_text(encoding="utf-8"))["resume_policy"] == "repair_required"
    # a control.json that appeared after the hold was read is never replaced, and then nothing else is
    # written (the watcher's own pause would otherwise hold its own stop)
    run3 = make_run(exp, "deploy3", [], switch=N)
    (run3 / "control.json").write_text(json.dumps({"cmd": "snapshot"}), encoding="utf-8")
    (saved / "supervisor_control.json").write_text(json.dumps({"cmd": "run"}), encoding="utf-8")
    stop = W.stop_run(run3, saved, "motor fixes kill line crossed: test")
    assert not stop["written"] and not stop["pause_written"] and not (run3 / "stop_reason.json").exists()
    assert json.loads((run3 / "control.json").read_text(encoding="utf-8")) == {"cmd": "snapshot"}
    assert json.loads((saved / "supervisor_control.json").read_text(encoding="utf-8")) == {"cmd": "run"}


def test_a_waiting_stop_is_printed_first_and_shown_on_the_dashboards(tmp_path, capsys):
    """The second review, finding 6: a waiting stop was reported only in the watcher's own log and state
    file. It now leads every poll's lines and --ensure, and the status file the dashboards read carries
    it (both dashboards name it)."""
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 1) + windows(N + 200, 3, **BAD), switch=N)
    saved = saved_for(tmp_path, run, control={"cmd": "pause", "reason": "an evaluation pause"})
    memory = W.Memory()
    _, first = W.watch_once(run, saved, memory=memory, poll_s=60.0)
    _, state = W.watch_once(run, saved, memory=memory, poll_s=60.0)
    assert state["phase"] == "waiting" and state["pending_stop"]["since"] == first["pending_stop"]["since"]
    W.print_state(state)
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("STOP WAITING") and "an evaluation pause" in out
    status = json.loads((saved / W.STATUS_NAME).read_text(encoding="utf-8"))
    assert status["phase"] == "waiting" and status["pending_stop"]["reason"].startswith("motor fixes")
    from mimo_bridge.tools.lineage_series import motor_watch_note
    note = motor_watch_note(saved, now=W.time.time() + 600)
    assert note.startswith("The motor fixes kill-line watcher found a crossing and has waited for 10 minutes")
    assert "a supervisor pause is in force" in note
    assert motor_watch_note(saved, run="another_run") is None and motor_watch_note(saved, run="deploy")
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    assert not bare_decimals(note), bare_decimals(note)
    # a watcher silent for five polls is named too
    assert "last polled" in motor_watch_note(saved, now=W.time.time() + 6 * 60 + 400)


# ------------------------------------------------------------------ the anchored eye line (the fourth review, finding 4)
LINEAGE_READINGS = [1.35, 3.15, 1.51, 0.21, 1.24, -0.22, 2.61, 0.97, 0.47, 0.87, 1.27]   # in 100, 380,550 to 414,200


def anchor_result(path, checkpoint, fraction):
    """A probe_saved_gaze.py / motor_fixes_eye_anchor.py result.json holding the anchored eye reading."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"checkpoint": str(checkpoint), "summary": {"saved_starts": {"50ms": {
        "visual_prediction_mse": {"error_reduction_fraction": fraction}}}}}), encoding="utf-8")
    return str(path)


def test_the_anchored_lines_distance_is_the_lineages_measured_noise():
    """ANCHOR_NOISE is the spread of the differences between successive readings of the scheduler's gaze test on
    the lineage (11 readings, 380,550 to 414,200: the difference of two readings 3,075 to 4,600 steps apart is what
    the line judges), and the line sits 1.96 of it under its reference: 2.5 healthy readings in 100 fall under."""
    d = np.diff(np.asarray(LINEAGE_READINGS) / 100.0)
    assert W.ANCHOR_NOISE == pytest.approx(float(np.std(d, ddof=1)), abs=5e-6)
    assert W.ANCHOR_Z == 1.96 and W.ANCHOR_DROP == pytest.approx(0.0307, abs=5e-5)


def _anchor_setup(tmp_path, before=0.012, floor=0.004):
    exp = tmp_path / "exp"
    run = make_run(exp, "deploy", windows(N + 100, 20), switch=N)
    saved = saved_for(tmp_path, run)
    refs = tmp_path / "refs"
    b = anchor_result(refs / "before" / "result.json", "C:/x/pre/checkpoints/ckpt_step_%d" % N, before)
    f = anchor_result(refs / "switched" / "result.json", "C:/x/pre/checkpoints/ckpt_step_%d" % N, floor)
    return exp, run, saved, dict(before=b, floor=f, folders=[str(tmp_path / "deploy_readings")])


def test_an_anchored_reading_under_the_line_is_a_stop_and_one_over_it_is_not(tmp_path):
    """Planted readings of a watched run's checkpoints: the line is min(before, switched) - 3.1 in 100; the
    scheduler's own gaze test result and the deploy's readings folder are both read; a reading of a checkpoint
    before N or of an unwatched run is not judged."""
    exp, run, saved, anchor = _anchor_setup(tmp_path)
    line = min(0.012, 0.004) - W.ANCHOR_DROP                                       # -0.0267
    sched = saved / "ability_rechecks"
    # the scheduler's test of a post-switch checkpoint, over the line: judged, no stop
    anchor_result(sched / ("gaze_prediction_%d_attempt1_1" % (N + 1500)) / "result.json",
                  run / "checkpoints" / ("ckpt_step_%d" % (N + 1500)), line + 0.001)
    # under the line, but of a checkpoint before the switch and of an unwatched run: not judged
    anchor_result(sched / ("gaze_prediction_%d_attempt1_2" % (N - 300)) / "result.json",
                  run / "checkpoints" / ("ckpt_step_%d" % (N - 300)), line - 0.05)
    anchor_result(sched / ("gaze_prediction_%d_attempt1_3" % (N + 1600)) / "result.json",
                  exp / "other_run" / "checkpoints" / ("ckpt_step_%d" % (N + 1600)), line - 0.05)
    why, state = W.watch_once(run, saved, anchor=anchor)
    assert why is None and not (run / "control.json").exists()
    a = state["anchor"]
    assert a["line"] == pytest.approx(line) and [r["step"] for r in a["readings"]] == [N + 1500]
    assert any(k.startswith(W.ANCHOR_KEY_PREFIX) for k in state["judged_blocks"])
    # the deploy's own reading of a later checkpoint, under the line: the stop, written into the active run
    anchor_result(tmp_path / "deploy_readings" / "after" / "result.json",
                  run / "checkpoints" / ("ckpt_step_%d" % (N + 1900)), line - 0.002)
    why, state = W.watch_once(run, saved, anchor=anchor)
    assert why and "anchored eye" in why and "checkpoint %d" % (N + 1900) in why
    assert "under the line of -2.7 in 100" in why and "before the switch, 1.2 in 100" in why
    assert json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    stop = [s for s in W.load_state(run)["stops"] if s.get("written")][0]
    assert stop["crossing_run"] == W.ANCHOR_CROSSING and stop["step"] == N + 1900
    # the reading that stopped is kept as judged, so a restarted watcher does not stop again on it
    assert any(k.endswith(str(Path(tmp_path / "deploy_readings" / "after" / "result.json").resolve()))
               for k in W.load_state(run)["judged_blocks"])
    assert W.ANCHOR_CROSSING not in W.Memory(W.load_state(run)).closed_runs()


def test_without_its_two_references_the_anchored_line_judges_nothing_and_says_so(tmp_path, capsys):
    exp, run, saved, anchor = _anchor_setup(tmp_path)
    anchor_result(saved / "ability_rechecks" / "gaze_prediction_1" / "result.json",
                  run / "checkpoints" / ("ckpt_step_%d" % (N + 1500)), -0.5)
    for missing in ("floor", "before"):
        a = dict(anchor, **{missing: None})
        why, state = W.watch_once(run, saved, anchor=a, memory=W.Memory(fresh=True))
        assert why is None and "line" not in state["anchor"] and not (run / "control.json").exists()
        assert "judges nothing" in state["anchor"]["why_blind"] and ("--anchor-" + missing) in state["anchor"]["why_blind"]
        W.print_state(state)
        assert "ANCHORED EYE LINE BLIND" in capsys.readouterr().out
    status = json.loads((saved / W.STATUS_NAME).read_text(encoding="utf-8"))
    assert "judges nothing" in status["anchored_eye"]["why_blind"]


def test_the_watch_awaits_the_anchored_reading_of_its_end(tmp_path):
    """With the anchored line the watch is complete only once a reading of a checkpoint at or after N + 6,000
    is judged (the procedure's end reading); without the line, the completion names it blind."""
    exp, run, saved, anchor = _anchor_setup(tmp_path)
    append(run, windows(N + 2100, 50))                       # windows to N + 7,000
    steps = np.arange(N, N + 7000)
    write_eye_chunks(run, steps, np.full(len(steps), 5.0), per_chunk=500)
    why, state = W.watch_once(run, saved, anchor=anchor)
    assert why is None and state["phase"] == "watching"
    assert "an anchored eye reading of a checkpoint at or after step %d" % (N + 6000) in state["watch_awaits"]
    anchor_result(tmp_path / "deploy_readings" / "end" / "result.json",
                  run / "checkpoints" / ("ckpt_step_%d" % (N + 6200)), 0.010)
    why, state = W.watch_once(run, saved, anchor=anchor)
    assert why is None and state["phase"] == "complete", state.get("watch_awaits")
    assert state["complete"]["anchored_eye"]["readings"][0]["step"] == N + 6200
    # the same watch without the references completes without the reading, and says the line judged nothing
    run2 = make_run(exp, "deploy_no_anchor", windows(N + 100, 70), switch=N)
    write_eye_chunks(run2, steps, np.full(len(steps), 5.0), per_chunk=500)
    why, state = W.watch_once(run2, saved_for(tmp_path, run2))
    assert state["phase"] == "complete" and "the anchored eye line judges nothing" in state["status"]
