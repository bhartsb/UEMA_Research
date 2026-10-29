# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Review probes for commit 8e7f6a5d9 (the training preflight's repair run in the relaunch script and the supervisor).

Written by the independent review of Oct 5, 2026 (docs/gain_of_function_v4/coordinator/
relaunch_repair_run_review_2026-10-05.md, branch review/relaunch-repair-run-2026-10-05, commit 64084e80a) and taken into
the suite by the coordinator with the review's fixes (its finding F5: the commit's own tests never ran main() for a
repair run). The relaunch script's main() is run end to end on temporary folders with every process call replaced:
psutil finds no process and refuses to end one, the trainer and the supervisor are stand-ins, the clock is a counter.
The watcher probes run the real hourly watcher on the written criteria. Nothing here looks at, ends or starts a real
trainer, supervisor or watcher, and nothing is written outside pytest's temporary folder.

The review's three strict-xfail probes (F1, F2, F3) pass on the fixes; the probes that recorded a defect as it stood
(the timeout that started no supervisor, the misspelled path that never stopped, the gates judged again that allowed the
start, the --set of --resume, the abbreviated flag) now assert the repaired behavior.
"""
import collections
import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tests.test_training_preflight import STOP_CRITERIA, _repair_setup
from mimo_bridge.tools import training_continuation as tc
from mimo_bridge.tools import training_preflight as TP

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py"
WATCHER = REPO / "docs/h84_launch_2026-09-18/hourly_watch.py"


def main_under_probe(tmp_path, monkeypatch, trainer="stats", supervisor_reports=True):
    """The relaunch script's main() wired to this test's preflight record (tests/test_training_preflight.py
    _repair_setup: a repair run of target gates 4, 5 and 7 from run_a's ckpt_step_1000, at most 2,000 model steps) and
    temporary Saved/ folders. trainer: "stats" writes a full stats record at once, "silent" never writes one and keeps
    running, "exits" ends at its first poll with exit code 1 before training, "trains_then_exits" writes a full stats
    record and ends at its first poll with exit code 0. supervisor_reports: whether the supervisor stand-in reports
    the run it was started for."""
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    spec = importlib.util.spec_from_file_location("relaunch_review_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    L = module.L
    saved = tmp_path / "saved"
    saved.mkdir()
    original = {"cmd": "pause", "reason": "held for the review probe", "archiver": "off"}
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(a), command=recorded)),
                                                      encoding="utf-8")
    (saved / "supervisor_control.json").write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(L, "SAVED", saved)
    monkeypatch.setattr(L, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(L, "RECEIPT", saved / "seg41_launch_receipt.json")
    monkeypatch.setattr(L, "stamp", lambda: "Oct 5, 2026, 6:00:00 PM PDT")
    monkeypatch.setattr(L, "trainers", lambda: [])
    monkeypatch.setattr(L, "guard", lambda: (True, {"committed_gb": 1.0}))
    monkeypatch.setattr(module, "newest_complete_checkpoint", lambda parent: start)
    monkeypatch.setattr(module, "manifest_refusal", lambda root=None: None)
    import psutil

    class NoProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, *x, **k):
            return []

        def terminate(self):
            raise AssertionError("the probe tried to end process %s" % self.pid)

        kill = terminate

    monkeypatch.setattr(psutil, "process_iter", lambda *x, **k: [])
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: ([], []))
    monkeypatch.setattr(psutil, "Process", NoProcess)
    plan0, refusal0, mark0, note0 = TP.launch_plan, TP.launch_refusal, TP.mark_launched, TP.note_attempt
    asked = []
    monkeypatch.setattr(TP, "launch_plan", lambda cmd, *x, **k: plan0(cmd, root=root, src=src))
    monkeypatch.setattr(TP, "launch_refusal", lambda cmd, *x, **k: asked.append([str(c) for c in cmd]) or refusal0(
        cmd, root=root, src=src, registry=reg))
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=root))
    monkeypatch.setattr(TP, "note_attempt", lambda f, r, by, *x, **k: note0(f, r, by, root=root))
    clock = {"now": 1_000_000.0}

    class Clock:
        @staticmethod
        def time():
            return clock["now"]

        @staticmethod
        def sleep(s):
            clock["now"] += s

        @staticmethod
        def strftime(fmt, *x):
            return time.strftime(fmt, *x)

    monkeypatch.setattr(module, "time", Clock())
    launched, started = {}, []

    class Trainer:
        pid = 4242
        returncode = None

        def poll(self):
            if trainer in ("exits", "trains_then_exits"):
                self.returncode = 1 if trainer == "exits" else 0
            return self.returncode

    def launch_trainer(cmd, run):
        launched.update(cmd=[str(c) for c in cmd], run=run, at=clock["now"],
                        criteria_before=(run / "declared_criteria.json").exists(),
                        control_before=json.loads((saved / "supervisor_control.json").read_text("utf-8")))
        (run / "load_report.json").write_text(json.dumps({"reinitialized_components": []}), encoding="utf-8")
        if trainer in ("stats", "trains_then_exits"):
            (run / "replay_training_stats.jsonl").write_text(json.dumps({"type": "mimo", "step": 1100}) + "\n",
                                                             encoding="utf-8")
        return Trainer()

    monkeypatch.setattr(L, "launch_trainer", launch_trainer)

    def popen(cmd, **kw):
        cmd = [str(c) for c in cmd]
        assert "mimo_bridge.tools.training_continuation" in cmd, cmd       # only the supervisor may be started
        if kw.get("stdout") is not None:
            kw["stdout"].close()
        started.append(dict(cmd=cmd, at=clock["now"], trainer_started=bool(launched),
                            control=json.loads((saved / "supervisor_control.json").read_text("utf-8"))))
        reported = cmd[cmd.index("--run") + 1] if supervisor_reports else str(tmp_path / "another_run")
        (saved / "training_continuation.json").write_text(json.dumps(dict(
            supervisor_pid=777, active_run=reported)), encoding="utf-8")
        return SimpleNamespace(pid=777)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(sys, "argv", ["relaunch.py", "--tag", "x", "--reason", "review probe"])
    assert str(L.SAVED).startswith(str(tmp_path)) and str(L.EXP).startswith(str(tmp_path))
    return module, SimpleNamespace(src=src, root=root, reg=reg, a=a, start=start, fp=fp, saved=saved,
                                   launched=launched, started=started, asked=asked, original=original, clock=clock)


def record(p):
    return TP._load(TP.record_path(p.fp, p.root))["repair_run"]


def control(p):
    return json.loads((p.saved / "supervisor_control.json").read_text(encoding="utf-8"))


def named(p, run):
    return control(p).get("cmd") == "pause" and control(p).get("repair_run") == str(run)


# ------------------------------------------------------------------ the repair run end to end
def test_main_launches_the_repair_run_bounded_judged_watched_and_used_once(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")
    module.main()
    run = p.launched["run"]
    flags = TP.flags_of(p.launched["cmd"])
    assert flags["--max-steps"] == "3000" and flags["--resume"] == str(p.start) and flags["--out"] == str(run)
    assert p.asked[-1] == p.launched["cmd"], "the preflight judged a command other than the one launched"
    assert p.launched["criteria_before"], "declared_criteria.json was not written before the trainer started"
    assert json.loads((run / "declared_criteria.json").read_text(encoding="utf-8")) == STOP_CRITERIA
    # the pause named the run before the trainer started (finding F1), keeping the control's other keys (F11)
    assert p.launched["control_before"]["repair_run"] == str(run) and p.launched["control_before"]["archiver"] == "off"
    assert record(p)["launched_at"] and record(p)["run"] == str(run)
    # the supervisor attached once, right after the trainer started, under the pause that names the run (F1)
    assert len(p.started) == 1 and "--replace-watcher" in p.started[0]["cmd"] and p.started[0]["trainer_started"]
    assert p.started[0]["at"] == p.launched["at"], "the supervisor waited for the first stats record"
    assert p.started[0]["control"]["cmd"] == "pause" and p.started[0]["control"]["repair_run"] == str(run)
    assert control(p) == p.started[0]["control"]                       # the pause stays after the attach
    assert tc.repair_run_pause(p.saved, run) is True and tc.pause_reason(p.saved, run) == "explicit supervisor pause"
    # the same record cannot start a second repair run, and normal training from it is refused
    again = TP.apply_sets(p.launched["cmd"], [("--max-steps", "3000")])
    assert TP.launch_plan(again, root=p.root, src=p.src)["kind"] == "normal"
    assert "never normal training" in TP.launch_refusal(again, root=p.root, src=p.src, registry=p.reg)


def test_an_early_exit_before_training_leaves_the_record_unused_and_the_run_paused(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "exits")
    with pytest.raises(SystemExit, match="exited early with 1"):
        module.main()
    run = p.launched["run"]
    assert not record(p)["launched_at"]                     # nothing trained: the run may be started again
    assert named(p, run) and len(p.started) == 1            # nothing relaunches it


def test_an_exit_after_training_past_the_start_marks_the_record_used(tmp_path, monkeypatch):
    # finding F6: a run that reached a short bound between two polls of the wait was left unused
    module, p = main_under_probe(tmp_path, monkeypatch, "trains_then_exits")
    with pytest.raises(SystemExit, match="exited early with 0"):
        module.main()
    assert record(p)["launched_at"] and record(p)["run"] == str(p.launched["run"])


# ------------------------------------------------------------------ the 30-minute wait that ends with the trainer running
def test_the_thirty_minute_timeout_marks_the_run_used_and_leaves_it_supervised_and_named(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "silent")
    with pytest.raises(SystemExit, match="runs under its supervisor and watcher"):
        module.main()
    run = p.launched["run"]
    assert record(p)["launched_at"]                         # used once
    assert len(p.started) == 1                              # its supervisor attached at the start
    assert named(p, run) and tc.repair_run_pause(p.saved, run) is True


def test_after_the_timeout_a_running_repair_trainer_is_named_by_the_pause_or_stopped(tmp_path, monkeypatch):
    # the review's strict-xfail probe of finding F1, passing on the fix
    module, p = main_under_probe(tmp_path, monkeypatch, "silent")
    with pytest.raises(SystemExit):
        module.main()
    run = p.launched["run"]
    stopped = (run / "control.json").exists() and json.loads((run / "control.json").read_text())["cmd"] in (
        "stop", "snapshot_and_stop")
    assert stopped or tc.repair_run_pause(p.saved, run) is True


def test_a_repair_trainer_whose_supervisor_does_not_attach_is_asked_to_save_and_stop(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "silent", supervisor_reports=False)
    with pytest.raises(SystemExit, match="did not report the new run"):
        module.main()
    run = p.launched["run"]
    assert json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    assert not record(p)["launched_at"] and named(p, run)


# ------------------------------------------------------------------ the pause as the pause watchdog reads it (F2)
def test_the_pause_watchdog_starts_nothing_while_the_repair_trainer_lives(tmp_path):
    from mimo_bridge.tools import pause_watchdog
    ctl = tmp_path / "supervisor_control.json"
    ctl.write_text(json.dumps({"cmd": "pause", "repair_run": str(tmp_path / "run"), "reason": "repair run"}),
                   encoding="utf-8")
    assert pause_watchdog.paused(ctl, alive=lambda run: True) is False
    assert pause_watchdog.run([sys.executable, "-c", "pass"], poll=0.2, control=ctl, alive=lambda run: True) == 4
    # after the repair trainer ends the pause holds training again: the measurements of its gates run then
    assert pause_watchdog.paused(ctl, alive=lambda run: False) is True
    assert pause_watchdog.run([sys.executable, "-c", "pass"], poll=0.2, control=ctl, alive=lambda run: False) == 0
    # with no live trainer of the named run in this test machine's process list
    assert pause_watchdog.paused(ctl) is True


def test_the_pause_watchdog_ends_a_command_when_the_repair_trainer_starts(tmp_path):
    from mimo_bridge.tools import pause_watchdog
    ctl = tmp_path / "supervisor_control.json"
    ctl.write_text(json.dumps({"cmd": "pause", "repair_run": str(tmp_path / "run")}), encoding="utf-8")
    seen = []

    def alive(run):                       # the repair trainer appears after the command has started
        seen.append(run)
        return len(seen) > 1
    started = time.time()
    code = pause_watchdog.run([sys.executable, "-c", "import time; time.sleep(30)"], poll=0.2, control=ctl,
                              alive=alive)
    assert code == 5 and time.time() - started < 20


def test_a_plain_pause_is_a_pause_whatever_runs(tmp_path):
    from mimo_bridge.tools import pause_watchdog
    ctl = tmp_path / "supervisor_control.json"
    ctl.write_text(json.dumps({"cmd": "pause", "reason": "held"}), encoding="utf-8")
    assert pause_watchdog.paused(ctl, alive=lambda run: pytest.fail("a plain pause names no run")) is True


# ------------------------------------------------------------------ stop criteria as the watcher judges them
HEALTHY_DISK = collections.namedtuple("usage", "total used free")(1e12, 0.0, 500e9)


def run_watcher(tmp_path, monkeypatch, criteria, vision, earlier=None, elapsed=3200):
    """The real hourly watcher on one stats record (the harness of tests/test_h84_hourly_watch.py), with `earlier`, a
    record written before it, when given: returns the control.json it wrote (None when it wrote none), the seconds it
    slept before it ended, and its stop reason."""
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: HEALTHY_DISK)
    from mimo_bridge import machine_stats
    from mimo_bridge.tools import commit_guard
    monkeypatch.setattr(machine_stats, "nvidia_smi_memory", lambda: (5000.0, 8188.0))
    monkeypatch.setattr(commit_guard, "commit_reading", lambda: (40.0, 68.8, 20.0))
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: collections.namedtuple("vm", "available")(64e9))
    monkeypatch.setenv("UEMA_WATCH_SAVED", str(tmp_path / "_saved"))
    monkeypatch.setenv("UEMA_WATCH_ARCHIVE", str(tmp_path / "_archive"))
    run = tmp_path / "run"
    run.mkdir()
    (run / "args.json").write_text("{}", encoding="utf-8")
    rec = dict(type="mimo", step=1500, timestamp=10000, elapsed_s=elapsed, nan_steps=0, model_steps_per_s=0.37,
               replay_loss=0.5, checkpoint_failures=0, peak_vram_mb=5200,
               token_temporal_std=dict(vision=.6, tactile=.6),
               causal_pe_per_modality=dict(vision=vision, tactile=.04, proprioception=.005),
               bout_body_speed=.003, follow_bout=.1, task_hand_fov=.05, cmd_response_cos_held=0.5)
    (run / "replay_training_stats.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ([earlier] if earlier else []) + [rec]), encoding="utf-8")
    (run / "declared_criteria.json").write_text(json.dumps(criteria), encoding="utf-8")
    slept = []
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(run), "2"])
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(time, "time", lambda: 10000)
    runpy.run_path(str(WATCHER), run_name="__main__")
    ctl = run / "control.json"
    reason = run / "stop_reason.json"
    return ((json.loads(ctl.read_text()) if ctl.exists() else None), sum(slept),
            (json.loads(reason.read_text())["reason"] if reason.exists() else None))


def test_a_declared_stop_is_judged_at_the_first_five_minute_check(tmp_path, monkeypatch):
    # finding F4: before the fix the first judgement came at the hourly sweep, 3,060 s after the watcher started
    stop, slept, why = run_watcher(tmp_path, monkeypatch, STOP_CRITERIA, vision=0.1)    # 0.1 is over the 0.04 line
    assert stop == {"cmd": "snapshot_and_stop"} and slept == 60 + 300 and "5-minute check" in why


def test_a_holding_stop_line_stops_nothing(tmp_path, monkeypatch):
    stop, slept, why = run_watcher(tmp_path, monkeypatch, STOP_CRITERIA, vision=0.01)
    assert stop is None and why is None


def test_a_stop_criterion_whose_field_is_absent_for_an_hour_of_records_stops_the_run(tmp_path, monkeypatch):
    # finding F3: a misspelled path printed "absent" every hour and never stopped the run, whatever the field read;
    # the re-check's finding N1: it stops once the run's records span a whole hour (here 3,800 s, to run-hour 1.9)
    typo = [dict(STOP_CRITERIA[0], path="causal_pe_per_modality.visoin")]
    assert TP.stop_criteria_problems(typo) == []                       # the form alone cannot see it
    earlier = dict(type="mimo", step=600, timestamp=6200, elapsed_s=3200, nan_steps=0)
    stop, slept, why = run_watcher(tmp_path, monkeypatch, typo, vision=0.1, earlier=dict(earlier), elapsed=7000)
    assert stop == {"cmd": "snapshot_and_stop"} and slept == 60 + 300 and "absent" in why and "visoin" in why


def test_a_stop_criterion_whose_field_is_absent_before_an_hour_of_records_stops_nothing(tmp_path, monkeypatch):
    # the re-check's finding N1: production writes some fields only from a segment's third or later record
    typo = [dict(STOP_CRITERIA[0], path="causal_pe_per_modality.visoin")]
    stop, slept, why = run_watcher(tmp_path, monkeypatch, typo, vision=0.1)        # one record, run-hour 0.9
    assert stop is None


def test_an_alert_whose_field_is_absent_stops_nothing(tmp_path, monkeypatch):
    alert = [dict(STOP_CRITERIA[1], path="cmd_response_cos_held_typo")]
    stop, slept, why = run_watcher(tmp_path, monkeypatch, alert, vision=0.1)
    assert stop is None


def test_a_stop_criterion_naming_a_field_the_stats_do_not_carry_is_refused(tmp_path):
    # the review's strict-xfail probe of finding F3, passing on the fix
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    (a / "replay_training_stats.jsonl").write_text(json.dumps(dict(
        type="mimo", step=1000, causal_pe_per_modality=dict(vision=0.01))) + "\n", encoding="utf-8")
    rec = TP._load(TP.record_path(fp, root))
    rec["repair_run"]["stop_criteria"] = [dict(STOP_CRITERIA[0], path="causal_pe_per_modality.visoin")]
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(TP.apply_sets(recorded, [("--max-steps", "3000")]), root=root, src=src, registry=reg)
    assert why and "visoin" in why


# ------------------------------------------------------------------ after the repair run (finding F9)
def test_target_gates_judged_again_allow_normal_training_only_from_a_state_added_after_the_launch(tmp_path):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    (tmp_path / "experiments/run_repair").mkdir(parents=True)        # marking needs the run folder (round two, R5)
    TP.mark_launched(fp, str(tmp_path / "experiments/run_repair"), "tester", root=root)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "PASS", "measured on the repair run's end checkpoint", ["row_%db" % n], "tester", root=root)
    assert TP.flags_of(recorded)["--resume"] == str(start)
    why = TP.launch_refusal(recorded, root=root, src=src, registry=reg)    # the start, whose weights never changed
    assert why and "ran its repair run" in why
    end = tmp_path / "experiments/run_repair/checkpoints/ckpt_step_3000"
    end.mkdir(parents=True)
    (tmp_path / "experiments/run_repair/args.json").write_text(json.dumps({"resume": str(start)}), encoding="utf-8")
    TP.add_state(fp, str(end), "gates 4, 5 and 7 judged again on it", "tester", root=root)
    cmd = list(recorded)
    cmd[cmd.index("--resume") + 1] = str(end)
    assert TP.launch_refusal(cmd, root=root, src=src, registry=reg) is None
    assert TP.launch_refusal(recorded, root=root, src=src, registry=reg)                   # the start: still refused


# ------------------------------------------------------------------ what the launch command can carry
def test_a_set_of_resume_or_out_is_refused(tmp_path, monkeypatch):
    # finding F7
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    spec = importlib.util.spec_from_file_location("relaunch_review_sets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.L, "EXP", tmp_path / "experiments")
    preflight = SimpleNamespace(launch_plan=lambda cmd: TP.launch_plan(cmd, root=root, src=src))
    run = module.run_folder("x")
    with pytest.raises(SystemExit, match="refused: --set --resume"):
        module.launch_command(recorded, a / "checkpoints/ckpt_step_1400", run, [("resume", str(start))], preflight)
    with pytest.raises(SystemExit, match="refused: --set --out"):
        module.launch_command(recorded, start, run, [("out", str(tmp_path / "elsewhere"))], preflight)


def test_an_abbreviated_flag_is_refused_before_it_can_undo_the_bound():
    # finding F8: what the trainer runs with, against what the preflight reads
    from mimo_bridge import train_mimo
    cmd = ["python", "-m", "mimo_bridge.train_mimo", "--out", "C:/runs/x", "--max-steps", "3000", "--max-step", "0"]
    assert TP.flags_of(cmd)["--max-steps"] == "3000"                    # what the bound check reads
    assert train_mimo.parse_args(cmd[3:]).max_steps == 0                # what the trainer would run with: no bound
    why = TP.launch_refusal(cmd)
    assert why.startswith("BLOCKED") and "the parser would read it as --max-steps" in why
