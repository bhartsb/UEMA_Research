# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tools import training_continuation as tc


@pytest.fixture(autouse=True)
def isolate_ability_worker(monkeypatch):
    # Ability-worker ownership is tested separately; supervisor unit tests never
    # inspect or launch the workspace's actual comparison scheduler.
    monkeypatch.setattr(tc,'ensure_ability_rechecks',lambda *a,**k:{'status':'test scheduler'})
    # likewise the developmental observer's digester (tests/test_observer_supervision.py)
    monkeypatch.setattr(tc,'ensure_observer',lambda *a,**k:{'status':'test observer'})
    # Synthetic checkpoints have no workspace launch authorization. Admission
    # behavior is exercised explicitly by the refusal test below.
    monkeypatch.setattr(tc,'resume_launch_refusal',lambda command:None)


def test_watcher_replacement_never_targets_its_own_supervisor():
    watcher = "C:/project/docs/hourly_watch.py"
    assert tc.is_watcher_command(["python", "-u", watcher, "run"])
    assert tc.is_watcher_command(["python", watcher, "run"])
    assert not tc.is_watcher_command(["python", "-u", "-m", "mimo_bridge.tools.training_continuation",
                                      "--watcher", watcher])


def checkpoint(run, step, complete=True):
    p = run / "checkpoints" / f"ckpt_step_{step}"
    p.mkdir(parents=True)
    (p / "meta.json").write_text(json.dumps({"step": step}))
    for name in ("model.safetensors", "optimizer.pt", "runtime_state.pt", "model_receipt.json"):
        if complete or name != "optimizer.pt":
            (p / name).write_text("{}" if name=='model_receipt.json' else "test")
    return p


def test_never_resume_partial_or_lexically_latest_checkpoint(tmp_path):
    expected = checkpoint(tmp_path, 99)
    checkpoint(tmp_path, 100, complete=False)
    checkpoint(tmp_path, 9)
    assert tc.latest_complete(tmp_path) == expected


def test_finished_stage_names_external_storage_and_ignores_stale_local_snapshot(tmp_path):
    run=tmp_path/'log_run';external=tmp_path/'other_drive'/'native_run'
    checkpoint(run,999)
    expected=checkpoint(external,101)
    checkpoint(external,102,complete=False)
    (run/'args.json').write_text(json.dumps({'checkpoint_storage':str(external/'checkpoints')}))
    record=tc.write_stage_done(run,ok=True,step=101,resumed_from='source',started_at=tc.time.time(),exception=None)
    assert Path(record['final_checkpoint'])==expected
    assert json.loads((run/'stage_done.json').read_text())==record
    assert record['steps']==101 and record['ok'] is True


def test_finished_stage_without_new_snapshot_does_not_claim_its_parent(tmp_path):
    run=tmp_path/'run';run.mkdir()
    record=tc.write_stage_done(run,ok=False,step=120,resumed_from='parent',started_at=tc.time.time(),exception='failed')
    assert record['final_checkpoint'] is None and record['exception']=='failed'


def test_required_episode_must_finish_before_checkpoint_is_selected(tmp_path):
    old=checkpoint(tmp_path,50);pending=checkpoint(tmp_path,60)
    (pending/'model_receipt.json').write_text(json.dumps({'episode_continuation':'episode_continuation_v1'}))
    assert tc.latest_complete(tmp_path)==old
    (pending/'episode_state.json').write_text('{}')
    assert tc.latest_complete(tmp_path)==pending


def test_crash_before_first_periodic_snapshot_can_resume_parent(tmp_path):
    parent = checkpoint(tmp_path / "parent", 400)
    child = tmp_path / "child"
    child.mkdir()
    command = ["python", "--resume", str(parent)]
    assert tc.resume_checkpoint(child, command) == parent
    newer = checkpoint(child, 500)
    assert tc.resume_checkpoint(child, command) == newer


def test_lock_excludes_second_supervisor(tmp_path):
    lock = tc.OwnerLock(tmp_path / "owner.lock")
    try:
        with pytest.raises(OSError):
            tc.OwnerLock(tmp_path / "owner.lock")
    finally:
        lock.close()
    tc.OwnerLock(tmp_path / "owner.lock").close()


def test_consumed_manual_stop_cannot_be_undone_by_supervisor(tmp_path):
    (tmp_path / "control_done_100.json").write_text('{"cmd":"snapshot"}')
    assert tc.pause_reason(tmp_path, tmp_path) is None
    (tmp_path / "control_done_101.json").write_text('{"cmd":"snapshot_and_stop"}')
    assert "deliberate stop" in tc.pause_reason(tmp_path, tmp_path)


@pytest.mark.parametrize("stop", [None, "manual", "repair", "other_trainer", "preflight"])
def test_exit_resumes_exact_flags_and_respects_stops(tmp_path, monkeypatch, stop):
    run = tmp_path / "run"
    ckpt = checkpoint(run, 555)
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run),
           "--resume", "old", "--hself-slots", "6400", "--audio-freq-bands", "off",
           "--fill-ring-first", "--babble-noise", "0.0"]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    calls = 0
    def processes(kind, requested=None):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return [process]  # attach
        if stop == "other_trainer" and requested is None:
            return [process]
        return []
    if stop == "manual":
        (root / "supervisor_control.json").write_text('{"cmd":"pause"}')
    elif stop == "repair":
        (run / "stop_reason.json").write_text('{"resume_policy":"repair_required","reason":"nonfinite"}')
    launches = []
    monkeypatch.setattr(tc, "processes", processes)
    monkeypatch.setattr(tc, "validate_checkpoint", lambda p: None)
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    if stop == "preflight":
        monkeypatch.setattr(tc, "resume_launch_refusal", lambda command: "changed source has no admitted record")
    # the archiver check is tested on its own below; here it must not scan this machine
    monkeypatch.setattr(tc, "ensure_archiver", lambda *a, **k: {"status": "running", "pids": [1]})
    class EndLoop(Exception):
        pass
    monkeypatch.setattr(tc.time, "sleep", lambda _: (_ for _ in ()).throw(EndLoop()))
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    assert len(launches) == (1 if stop is None else 0)
    if stop == "preflight":
        status = json.loads((root / "training_continuation.json").read_text())
        assert status["status"] == "continuation preflight refused"
        assert status["preflight_refusal"] == "changed source has no admitted record"
    if launches:
        sent = launches[0]
        assert tc.argument(sent, "--resume") == str(ckpt)
        assert tc.argument(sent, "--audio-freq-bands") == "off"
        assert tc.argument(sent, "--hself-slots") == "6400"
        assert tc.argument(sent, "--babble-noise") == "0.0"
        assert "--fill-ring-first" in sent
        assert tc.argument(sent, "--out") != str(run)


def test_a_second_resume_from_the_same_checkpoint_pauses_for_repair(tmp_path, monkeypatch):
    """Measured 2026-09-20: H84 crashed on a runaway self-percept and this supervisor
    resumed it 18 times from ckpt_step_231473, each attempt living about 20 minutes - past
    the 10-minute healthy-start reset - and reaching no new checkpoint. 6.4 hours of wall
    clock bought 418 steps. Resuming twice from the SAME checkpoint means the crash
    repeats, so the second time pauses for repair instead of launching."""
    run = tmp_path / "run"
    ckpt = checkpoint(run, 555)
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run), "--resume", "old"]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    calls = 0
    def processes(kind, requested=None):
        nonlocal calls
        calls += 1
        return [process] if calls <= 2 else []       # attach, then no trainer ever again
    launches = []
    monkeypatch.setattr(tc, "processes", processes)
    monkeypatch.setattr(tc, "validate_checkpoint", lambda p: None)
    monkeypatch.setattr(tc, "validated_resume_checkpoint", lambda r, c: (ckpt, []))
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    monkeypatch.setattr(tc, "ensure_archiver", lambda *a, **k: {"status": "running", "pids": [1]})
    sweeps = 0
    class EndLoop(Exception):
        pass
    def sleep(_):
        nonlocal sweeps
        sweeps += 1
        if sweeps >= 3:
            raise EndLoop()
        # the relaunch gate only opens 60 s after the previous start
        monkeypatch.setattr(tc.time, "time", lambda base=tc.time.time(): base + 600 * sweeps)
    monkeypatch.setattr(tc.time, "sleep", sleep)
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    assert len(launches) == 1, 'the same checkpoint was resumed from more than once'
    status = json.loads((root / "training_continuation.json").read_text(encoding="utf-8"))["status"]
    assert 'repair' in status and 'same checkpoint' in status, status


@pytest.mark.parametrize("control, watcher_started", [
    ({"cmd": "pause", "repair_run": "RUN"}, True),           # the relaunch script's pause for a repair run
    ({"cmd": "pause"}, False),                               # any other pause: as before, no watcher is started
    ({"cmd": "pause", "repair_run": "OTHER"}, False),        # a pause naming another run
    ({"cmd": "run"}, True)])
def test_a_repair_run_pause_keeps_its_watcher_and_relaunches_nothing(tmp_path, monkeypatch, control, watcher_started):
    """The training preflight's repair run (the operator, Oct 5, 2026) cannot continue into normal training by itself:
    the relaunch script leaves supervisor_control.json a pause naming the run. Before Oct 5 a pause also kept the
    supervisor from starting the run's hourly watcher, so a repair run would have trained with no stop rule judged."""
    run = tmp_path / "run"
    checkpoint(run, 555)
    root = tmp_path / "supervision"
    root.mkdir()
    named = {"RUN": str(run), "OTHER": str(tmp_path / "other")}
    (root / "supervisor_control.json").write_text(json.dumps(
        dict(control, **({"repair_run": named[control["repair_run"]]} if "repair_run" in control else {}))))
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run), "--resume", "old", "--max-steps", "3000"]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    alive = True
    def processes(kind, requested=None):
        return [process] if kind == "trainer" and alive else []
    launches = []
    monkeypatch.setattr(tc, "processes", processes)
    monkeypatch.setattr(tc, "validate_checkpoint", lambda p: None)
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append([str(x) for x in c]))
    monkeypatch.setattr(tc, "ensure_archiver", lambda *a, **k: {"status": "running", "pids": [1]})
    sweeps = 0
    class EndLoop(Exception):
        pass
    def sleep(_):
        nonlocal sweeps, alive
        sweeps += 1
        alive = False                                    # the trainer ends at its bound after the first sweep
        if sweeps >= 3:
            raise EndLoop()
        monkeypatch.setattr(tc.time, "time", lambda base=tc.time.time(): base + 600 * sweeps)
    monkeypatch.setattr(tc.time, "sleep", sleep)
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    watchers = [c for c in launches if str(tmp_path / "hourly_watch.py") in c]
    trainers = [c for c in launches if "mimo_bridge.train_mimo" in c]
    assert len(watchers) == (1 if watcher_started else 0)
    assert tc.repair_run_pause(root, run) is (control.get("repair_run") == "RUN")
    if control["cmd"] == "pause":
        assert trainers == [], "a paused supervisor continued the run"
        status = json.loads((root / "training_continuation.json").read_text(encoding="utf-8"))["status"]
        assert status == "paused for repair or explicit instruction", status
    # a run whose watcher already wrote its stop gets no new watcher
    (run / "stop_reason.json").write_text('{"resume_policy": "repair_required", "reason": "a stop line"}')
    assert tc.repair_run_pause(root, run) is False


def test_a_newer_checkpoint_in_the_started_run_is_progress(tmp_path):
    run = tmp_path / "run"
    checkpoint(run, 555)
    assert not tc.newer_checkpoint(run, str(tmp_path / "parent" / "ckpt_step_555"))
    checkpoint(run, 600, complete=False)
    assert not tc.newer_checkpoint(run, str(tmp_path / "parent" / "ckpt_step_555")), "an incomplete save is no progress"
    checkpoint(run, 601)
    assert tc.newer_checkpoint(run, str(tmp_path / "parent" / "ckpt_step_555"))
    assert not tc.newer_checkpoint(run, None)


def test_planned_stops_that_each_saved_a_newer_checkpoint_never_read_as_failed_starts(tmp_path, monkeypatch):
    """2026-09-24: relaunches at 8:34, 9:06 and 9:24 PM PDT were each stopped within 5 minutes, with a
    new checkpoint, by comparison windows and evaluation pauses; after the 9:28 PM stop the
    supervisor refused to resume ("three failed starts"). A start that saved a newer checkpoint
    made progress, so the supervisor keeps resuming."""
    run = tmp_path / "run"
    checkpoint(run, 555)
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run), "--resume", "old"]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    calls = 0
    def processes(kind, requested=None):
        nonlocal calls
        calls += 1
        return [process] if calls <= 2 else []       # attach, then every started trainer has stopped
    launches = []
    def launch(c, cwd, log):
        launches.append(c)
        checkpoint(Path(tc.argument(c, "--out")), 555 + 100 * len(launches))   # saved, then stopped
    names = iter(range(1000))
    monkeypatch.setattr(tc.time, "strftime", lambda fmt, *a: "t%03d" % next(names))
    monkeypatch.setattr(tc, "processes", processes)
    monkeypatch.setattr(tc, "validate_checkpoint", lambda p: None)
    monkeypatch.setattr(tc, "launch", launch)
    monkeypatch.setattr(tc, "stop_watchers", lambda run: None)
    monkeypatch.setattr(tc, "ensure_archiver", lambda *a, **k: {"status": "running", "pids": [1]})
    sweeps = 0
    class EndLoop(Exception):
        pass
    base = tc.time.time()
    def sleep(_):
        nonlocal sweeps
        sweeps += 1
        if sweeps >= 7:
            raise EndLoop()
        # 120 s per sweep: past the 60 s relaunch gate, far under the 10-minute healthy-start reset
        monkeypatch.setattr(tc.time, "time", lambda s=sweeps: base + 120 * s)
    monkeypatch.setattr(tc.time, "sleep", sleep)
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    assert len(launches) >= 5, "only %d starts: planned stops read as failed starts" % len(launches)
    status = json.loads((root / "training_continuation.json").read_text(encoding="utf-8"))["status"]
    assert "failed starts" not in status, status


def test_three_starts_that_saved_nothing_new_still_pause_for_repair(tmp_path, monkeypatch):
    """The failed-start guard still holds when no start makes progress (each resumes from a different
    earlier checkpoint and saves nothing)."""
    run = tmp_path / "run"
    checkpoints = [checkpoint(tmp_path / "parents", s) for s in (300, 400, 500, 600)]
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run), "--resume", "old"]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    calls = 0
    def processes(kind, requested=None):
        nonlocal calls
        calls += 1
        return [process] if calls <= 2 else []
    launches = []
    offered = iter(checkpoints)
    monkeypatch.setattr(tc, "validated_resume_checkpoint", lambda r, c: (next(offered), []))
    names = iter(range(1000))
    monkeypatch.setattr(tc.time, "strftime", lambda fmt, *a: "t%03d" % next(names))
    monkeypatch.setattr(tc, "processes", processes)
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    monkeypatch.setattr(tc, "stop_watchers", lambda run: None)
    monkeypatch.setattr(tc, "ensure_archiver", lambda *a, **k: {"status": "running", "pids": [1]})
    sweeps = 0
    class EndLoop(Exception):
        pass
    base = tc.time.time()
    def sleep(_):
        nonlocal sweeps
        sweeps += 1
        if sweeps >= 6:
            raise EndLoop()
        monkeypatch.setattr(tc.time, "time", lambda s=sweeps: base + 120 * s)
    monkeypatch.setattr(tc.time, "sleep", sleep)
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    assert len(launches) == 3
    status = json.loads((root / "training_continuation.json").read_text(encoding="utf-8"))["status"]
    assert "three failed starts" in status, status


# ------------------------------------------------------------------ exactly one checkpoint archiver
def test_a_running_archiver_is_left_alone(tmp_path, monkeypatch):
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [4242])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    assert tc.ensure_archiver(tmp_path, tmp_path) == {"status": "running", "pids": [4242],
                                                      "lineage": None}
    assert launches == []


def test_with_no_archiver_one_is_started_with_the_watching_command(tmp_path, monkeypatch):
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append((c, log)))
    got = tc.ensure_archiver(tmp_path, tmp_path, now=1000.0)
    assert got["started_at"] == 1000.0 and len(launches) == 1
    cmd, log = launches[0]
    assert cmd[1:] == list(tc.ARCHIVER_COMMAND) and "--watch" in cmd and "--dry-run" not in cmd
    assert Path(log).name == "checkpoint_archive_stdout.log"
    from mimo_bridge.tools.archive_checkpoints import is_archiver_command
    assert is_archiver_command(cmd), "the supervisor would not recognise its own archiver"


def test_a_held_archiver_lock_means_no_second_archiver(tmp_path, monkeypatch):
    from mimo_bridge.tools.archive_checkpoints import LOCK_NAME
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    held = tc.OwnerLock(tmp_path / LOCK_NAME)
    try:
        assert "lock is held" in tc.ensure_archiver(tmp_path, tmp_path)["status"]
    finally:
        held.close()
    assert launches == []


def test_an_archiver_that_dies_is_restarted_at_most_once_per_retry_window(tmp_path, monkeypatch):
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    assert "next start allowed" in tc.ensure_archiver(tmp_path, tmp_path, last_start=1000.0,
                                                      now=1000.0 + 60)["status"]
    assert launches == []
    tc.ensure_archiver(tmp_path, tmp_path, last_start=1000.0, now=1000.0 + tc.ARCHIVER_RETRY_S)
    assert len(launches) == 1


def test_the_supervisor_checks_the_archiver_on_every_sweep_and_reports_it(tmp_path, monkeypatch):
    run = tmp_path / "run"
    checkpoint(run, 555)
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run)]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    monkeypatch.setattr(tc, "processes", lambda kind, requested=None: [process] if kind == "trainer" else [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: None)
    calls = []

    def fake(root_, cwd, last_start=None, now=None, lineage=None):
        calls.append(last_start)
        return {"status": "started (none was running)", "started_at": 5.0} if not calls[:-1] else {"status": "running", "pids": [9]}
    monkeypatch.setattr(tc, "ensure_archiver", fake)
    sweeps = 0

    class EndLoop(Exception):
        pass

    def sleep(_):
        nonlocal sweeps
        sweeps += 1
        if sweeps >= 2:
            raise EndLoop()
    monkeypatch.setattr(tc.time, "sleep", sleep)
    (run / "replay_training_stats.jsonl").write_text("{}")
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    assert calls == [None, 5.0], "the start time of a launched archiver was not carried to the next sweep"
    state = json.loads((root / "training_continuation.json").read_text(encoding="utf-8"))
    assert state["archiver"] == {"status": "running", "pids": [9]}


def test_a_failing_archiver_check_never_stops_the_supervisor(tmp_path, monkeypatch):
    run = tmp_path / "run"
    checkpoint(run, 555)
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run)]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    monkeypatch.setattr(tc, "processes", lambda kind, requested=None: [process] if kind == "trainer" else [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: None)

    def broken(*a, **k):
        raise RuntimeError("psutil exploded")
    monkeypatch.setattr(tc, "ensure_archiver", broken)

    class EndLoop(Exception):
        pass
    monkeypatch.setattr(tc.time, "sleep", lambda _: (_ for _ in ()).throw(EndLoop()))
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    state = json.loads((root / "training_continuation.json").read_text(encoding="utf-8"))
    assert "check failed" in state["archiver"]["status"] and "psutil exploded" in state["archiver"]["status"]


# ------------------------------------------------------------ the archiver's lineage and off switch
def _lineage(tmp_path):
    """root <- seg2 <- seg3, each segment resuming the one before by its checkpoint."""
    exp = tmp_path / "experiments"
    root = exp / "H99_lineage"
    c1 = checkpoint(root, 100)
    (root / "args.json").write_text(json.dumps({"resume": None}))
    seg2 = exp / "H99_lineage_seg2"
    c2 = checkpoint(seg2, 200)
    (seg2 / "args.json").write_text(json.dumps({"resume": str(c1)}))
    seg3 = exp / "H99_lineage_seg3"
    checkpoint(seg3, 300)
    (seg3 / "args.json").write_text(json.dumps({"resume": str(c2)}))
    return seg3


def test_the_archiver_is_started_for_the_held_runs_lineage(tmp_path, monkeypatch):
    """Review ops-storage finding 9: the command carried no --lineage, so every supervisor
    started H84's archiver whatever it held."""
    seg3 = _lineage(tmp_path)
    assert tc.lineage_of(seg3) == "H99_lineage"
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    got = tc.ensure_archiver(tmp_path, tmp_path, now=1.0, lineage=tc.lineage_of(seg3))
    assert got["lineage"] == "H99_lineage"
    cmd = launches[0]
    assert cmd[cmd.index("--lineage") + 1] == "H99_lineage"
    from mimo_bridge.tools.archive_checkpoints import is_archiver_command
    assert is_archiver_command(cmd)


def test_the_supervisor_passes_the_lineage_of_the_run_it_holds(tmp_path, monkeypatch):
    run = _lineage(tmp_path)
    root = tmp_path / "supervision"
    root.mkdir()
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", str(run)]
    process = SimpleNamespace(pid=100, info={"cmdline": cmd})
    monkeypatch.setattr(tc, "processes", lambda kind, requested=None: [process] if kind == "trainer" else [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: None)
    seen = []
    monkeypatch.setattr(tc, "ensure_archiver",
                        lambda *a, lineage=None, **k: seen.append(lineage) or {"status": "running"})

    class EndLoop(Exception):
        pass
    monkeypatch.setattr(tc.time, "sleep", lambda _: (_ for _ in ()).throw(EndLoop()))
    (run / "replay_training_stats.jsonl").write_text("{}")
    with pytest.raises(EndLoop):
        tc.supervise(run, root, tmp_path / "hourly_watch.py")
    assert seen == ["H99_lineage"]


def test_archiver_off_in_the_control_file_starts_none_and_never_lifts_a_pause(tmp_path, monkeypatch):
    """verify-fix-ops 2026-09-23: the pause and the archiver switch share supervisor_control.json, so
    {"archiver": "off"} written over {"cmd": "pause"} read as "not paused" and a new-code supervisor
    would have relaunched training in the middle of a paused build. A file without cmd keeps training
    paused (as it did before the switch existed); {"cmd": "run", "archiver": "off"} runs it."""
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"archiver": "off"}))
    got = tc.ensure_archiver(tmp_path, tmp_path, now=1.0, lineage="H99_lineage")
    assert got["status"].startswith("off") and launches == []
    why = tc.pause_reason(tmp_path, tmp_path)
    assert why and "no cmd" in why and "stays paused" in why, why
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"cmd": "run", "archiver": "off"}))
    assert tc.pause_reason(tmp_path, tmp_path) is None, "an archiver setting beside a cmd paused training"
    assert tc.ensure_archiver(tmp_path, tmp_path, now=1.0, lineage="H99_lineage")["status"].startswith("off")
    assert launches == []
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"cmd": "pause", "archiver": "off"}))
    assert tc.pause_reason(tmp_path, tmp_path) == "explicit supervisor pause"
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"something": "else"}))
    assert tc.pause_reason(tmp_path, tmp_path) == "unreadable supervisor control"
    # the control: with the entry gone, one is started
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"cmd": "run"}))
    tc.ensure_archiver(tmp_path, tmp_path, now=1.0, lineage="H99_lineage")
    assert len(launches) == 1


def test_a_running_archiver_for_another_lineage_is_named_and_left_running(tmp_path, monkeypatch):
    from mimo_bridge.tools.archive_checkpoints import STATUS_NAME
    launches = []
    monkeypatch.setattr(tc, "archiver_pids", lambda: [4242])
    monkeypatch.setattr(tc, "launch", lambda c, cwd, log: launches.append(c))
    (tmp_path / STATUS_NAME).write_text(json.dumps({"lineage": "H84_fresh_drivers_20260918"}))
    got = tc.ensure_archiver(tmp_path, tmp_path, lineage="H99_lineage")
    assert "for lineage H84_fresh_drivers_20260918" in got["status"] and launches == []


def test_a_moved_resume_checkpoint_is_resolved_to_its_archive_copy(tmp_path, monkeypatch):
    """Review ops-storage finding 11: train_mimo opens --resume before anything reads the move
    records, so the supervisor must hand it the archive path."""
    from mimo_bridge import storage_locations as SL
    c_ck = tmp_path / "C" / "run" / "checkpoints" / "ckpt_step_400"
    d_ck = checkpoint(tmp_path / "D" / "run", 400)
    manifest = tmp_path / "C" / "storage_moves.jsonl"
    SL.append_row(manifest, dict(event="moved", source=str(c_ck), destination=str(d_ck),
                                 files={"meta.json": SL.sha256_file(d_ck / "meta.json")}))
    monkeypatch.setattr(SL, "C_MANIFEST", manifest)
    monkeypatch.setattr(SL, "D_MANIFESTS", ())
    monkeypatch.setattr(tc, "validate_checkpoint", lambda p: None)
    child = tmp_path / "child"
    child.mkdir()
    command = ["python", "--resume", str(c_ck)]
    got, rejected = tc.validated_resume_checkpoint(child, command)
    assert got == d_ck and rejected == []
    assert tc.resume_checkpoint(child, command) == d_ck
    # a move row whose copy is gone is rejected with its reason, never a crash of the loop
    import shutil
    shutil.rmtree(d_ck)
    got, rejected = tc.validated_resume_checkpoint(child, command)
    assert got is None and "nothing is there" in rejected[0]["error"]
