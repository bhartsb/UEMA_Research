# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The relaunch script's part of the training preflight's CONTINUATION (the operator's plan of Oct 11, 2026, "Prepare one
complete launch decision": "The current repair route has already used its allowed run. Prepare a narrowly named
continuation route, preserving the other holds; do not simply remove that restriction."). The script treats a
continuation as it treats a repair run: --max-steps reads the bound, the stop criteria reach the hourly watcher as
declared criteria before the trainer starts, the supervisor is left a pause naming the run, the attempt is noted and the
launch marked in the record's continuation block (never in its used repair run's block), and a second use is refused by
the preflight. main() runs on temporary folders with every process call replaced (the harness of
tests/test_relaunch_repair_run_review.py); nothing here looks at, ends or starts a real trainer, supervisor or watcher,
and nothing is written outside pytest's temporary folder."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tests.test_relaunch_repair_run_review import SCRIPT
from mimo_bridge.tests.test_training_preflight import STOP_CRITERIA, _continuation_block, _continuation_setup
from mimo_bridge.tools import training_continuation as tc
from mimo_bridge.tools import training_preflight as TP


def _module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main_under_probe(tmp_path, monkeypatch, trainer="stats"):
    """The relaunch script's main() wired to tests/test_training_preflight.py _continuation_setup (the repair run of
    target gates 4, 5 and 7 launched from run_a's ckpt_step_1000 and completed in run_repair at 3,000; its continuation
    from run_repair's ckpt_step_3000, at most 1,000 model steps) and temporary Saved/ folders, the supervisor's record
    naming run_repair with the continuation's command. trainer: "stats" writes a full stats record at once, "exits"
    ends at its first poll with exit code 1 before training, "trains_then_exits" writes a full stats record and ends at
    its first poll with exit code 0."""
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    module = _module("relaunch_continuation_probe")
    L = module.L
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(rep), command=cont)),
                                                      encoding="utf-8")
    (saved / "supervisor_control.json").write_text(json.dumps({"cmd": "pause", "reason": "held", "archiver": "off"}),
                                                   encoding="utf-8")
    monkeypatch.setattr(L, "SAVED", saved)
    monkeypatch.setattr(L, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(L, "RECEIPT", saved / "seg41_launch_receipt.json")
    monkeypatch.setattr(L, "stamp", lambda: "Oct 11, 2026, 9:00:00 AM PDT")
    monkeypatch.setattr(L, "trainers", lambda: [])
    monkeypatch.setattr(L, "guard", lambda: (True, {"committed_gb": 1.0}))
    monkeypatch.setattr(module, "newest_complete_checkpoint", lambda parent: end)
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
    asked, marks = [], []
    monkeypatch.setattr(TP, "launch_plan", lambda cmd, *x, **k: plan0(cmd, root=root, src=src))
    monkeypatch.setattr(TP, "launch_refusal", lambda cmd, *x, **k: asked.append(
        (refusal0(cmd, root=root, src=src, registry=reg), [str(c) for c in cmd])) or asked[-1][0])
    # the block the script names reaches the record (the repair run's harness drops it)
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: marks.append(("mark", k)) or mark0(
        f, r, by, root=root, **k))
    monkeypatch.setattr(TP, "note_attempt", lambda f, r, by, *x, **k: marks.append(("note", k)) or note0(
        f, r, by, root=root, **k))
    clock = {"now": 3_000_000.0}

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
    launched, started = [], []
    mode = {"trainer": trainer}                     # a test may change it between two launches

    class Trainer:
        pid = 4242
        returncode = None

        def __init__(self):
            self.mode = mode["trainer"]

        def poll(self):
            if self.mode in ("exits", "trains_then_exits"):
                self.returncode = 1 if self.mode == "exits" else 0
            return self.returncode

    def launch_trainer(cmd, run):
        launched.append(dict(cmd=[str(c) for c in cmd], run=run,
                             criteria_before=(run / "declared_criteria.json").exists(),
                             control_before=json.loads((saved / "supervisor_control.json").read_text("utf-8")),
                             record_before=TP._load(TP.record_path(fp, root))))
        (run / "load_report.json").write_text(json.dumps({"reinitialized_components": []}), encoding="utf-8")
        # the latent body shadow creates its output folder at the trainer's first model step, before the first full
        # stats record (mimo_bridge/latent_body_runtime.py; the re-check of Oct 11, 2026)
        output = TP.flags_of(cmd).get("--latent-body-output")
        if isinstance(output, str):
            Path(output).mkdir(parents=True, exist_ok=True)
        trainer = mode["trainer"]
        if trainer in ("stats", "trains_then_exits"):
            (run / "replay_training_stats.jsonl").write_text(json.dumps({"type": "mimo", "step": 3100}) + "\n",
                                                             encoding="utf-8")
        return Trainer()

    monkeypatch.setattr(L, "launch_trainer", launch_trainer)

    def popen(cmd, **kw):
        cmd = [str(c) for c in cmd]
        assert "mimo_bridge.tools.training_continuation" in cmd, cmd       # only the supervisor may be started
        if kw.get("stdout") is not None:
            kw["stdout"].close()
        started.append(dict(cmd=cmd, control=json.loads((saved / "supervisor_control.json").read_text("utf-8"))))
        (saved / "training_continuation.json").write_text(json.dumps(dict(
            supervisor_pid=777, active_run=cmd[cmd.index("--run") + 1], command=[sys.executable] + launched[-1]["cmd"][
                1:])), encoding="utf-8")
        return SimpleNamespace(pid=777)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    assert str(L.SAVED).startswith(str(tmp_path)) and str(L.EXP).startswith(str(tmp_path))
    return module, SimpleNamespace(src=src, root=root, reg=reg, start=start, rep=rep, end=end, cont=cont, fp=fp,
                                   saved=saved, launched=launched, started=started, asked=asked, marks=marks,
                                   mode=mode)


def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["relaunch.py", *argv])
    module.main()


def record(p):
    return TP._load(TP.record_path(p.fp, p.root))


def test_main_launches_the_continuation_bounded_watched_paused_and_used_once(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")
    repair_before = record(p)["repair_run"]
    run_main(module, monkeypatch, "--tag", "x", "--reason", "the continuation to 4,000")
    assert len(p.launched) == 1
    run = p.launched[0]["run"]
    flags = TP.flags_of(p.launched[0]["cmd"])
    assert flags["--max-steps"] == "4000" and flags["--resume"] == str(p.end) and flags["--out"] == str(run)
    # the block's run flags reach the trainer as written (the review of Oct 11, 2026, finding F5)
    run_flags = record(p)["continuation"]["run_flags"]
    assert run_flags and all(flags[k] == v for k, v in run_flags.items())
    assert p.asked[-1] == (None, p.launched[0]["cmd"]), "the preflight judged a command other than the one launched"
    # the stop criteria and the pause naming the run were written before the trainer started, the control's other keys
    # kept; the attempt was noted in the continuation block before the trainer started
    assert p.launched[0]["criteria_before"]
    assert json.loads((run / "declared_criteria.json").read_text(encoding="utf-8")) == STOP_CRITERIA
    before = p.launched[0]["control_before"]
    assert before["cmd"] == "pause" and before["repair_run"] == str(run) and before["archiver"] == "off"
    assert "continuation of configuration" in before["reason"] and "model step 4000" in before["reason"]
    assert [a["run"] for a in p.launched[0]["record_before"]["continuation"]["attempts"]] == [str(Path(run).resolve())]
    assert [m for m in p.marks] == [("note", {"block": "continuation"}), ("mark", {"block": "continuation"})]
    # the supervisor attached once under that pause, which stays; nothing relaunches the run
    assert len(p.started) == 1 and p.started[0]["control"]["repair_run"] == str(run)
    assert tc.repair_run_pause(p.saved, run) is True and tc.pause_reason(p.saved, run) == "explicit supervisor pause"
    # marked in the continuation block; the used repair run's block is kept as written
    rec = record(p)
    assert rec["continuation"]["launched_at"] and rec["continuation"]["run"] == str(Path(run).resolve())
    assert rec["repair_run"] == repair_before
    # a second use is normal training, refused by the preflight before anything is ended or started
    saved_control = p.saved / "supervisor_control.json"
    saved_control.write_text(json.dumps({"cmd": "pause", "reason": "held"}), encoding="utf-8")
    (p.saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(p.rep), command=p.cont)),
                                                        encoding="utf-8")
    with pytest.raises(SystemExit, match="refused by the training preflight"):
        run_main(module, monkeypatch, "--tag", "y", "--reason", "a second continuation")
    assert len(p.launched) == 1 and json.loads(saved_control.read_text(encoding="utf-8"))["reason"] == "held"
    why = p.asked[-1][0]
    assert why and "3 of 7 gates block launch" in why and TP.flags_of(p.asked[-1][1])["--max-steps"] == "0"


def test_an_early_exit_before_training_leaves_the_continuation_unused(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "exits")
    with pytest.raises(SystemExit, match="exited early with 1"):
        run_main(module, monkeypatch, "--tag", "x", "--reason", "the continuation")
    assert not record(p)["continuation"]["launched_at"]
    assert TP.continuation_available(record(p)["continuation"])
    assert tc.repair_run_pause(p.saved, p.launched[0]["run"]) is True


def _retry_block(p, output):
    """The continuation block written again with a new --latent-body-output (the rest as _continuation_setup wrote it)."""
    flags = dict(record(p)["continuation"]["run_flags"], **{"--latent-body-output": str(output)})
    return _continuation_block(p.end, run_flags=flags)


def test_a_trainer_that_dies_before_training_leaves_a_retry_with_a_new_latent_output_folder(tmp_path, monkeypatch):
    # the re-check of Oct 11, 2026: the attempt is noted, the shadow creates its output folder at the first model step,
    # the trainer dies before its first full stats record; the retry as written is refused (the folder exists), the
    # block is written again with a new --latent-body-output (the attempt carried into it) and the retry launches
    module, p = main_under_probe(tmp_path, monkeypatch, "exits")
    first_output = record(p)["continuation"]["run_flags"]["--latent-body-output"]
    with pytest.raises(SystemExit) as exc:
        run_main(module, monkeypatch, "--tag", "x", "--reason", "the continuation")
    text, run1 = str(exc.value), p.launched[0]["run"]
    # the early exit names the recorded attempt, its run folder, its latent output folder and the rewrite needed
    assert text.startswith("relaunched trainer exited early with 1") and "stays unused" in text and str(run1) in text
    assert ("latent body output folder %s exists;" % first_output) in text and "new run_flags --latent-body-output" in text
    assert Path(first_output).is_dir() and TP.continuation_available(record(p)["continuation"])
    # the retry as written: refused before anything is ended or started
    with pytest.raises(SystemExit, match="refused by the training preflight"):
        run_main(module, monkeypatch, "--tag", "y", "--reason", "the retry as written")
    assert "the latent body output folder %s exists already" % first_output in p.asked[-1][0]
    assert len(p.launched) == 1
    # the block written again with a new output folder; the retry carries the same flag and launches
    retry_output = tmp_path / "latent" / "continuation_retry"
    TP.set_continuation(p.fp, _retry_block(p, retry_output), "tester", root=p.root, registry=p.reg)
    assert [a["run"] for a in record(p)["continuation"]["attempts"]] == [str(Path(run1).resolve())]
    p.mode["trainer"] = "stats"
    run_main(module, monkeypatch, "--tag", "z", "--reason", "the retry", "--set", "latent-body-output",
             str(retry_output))
    assert p.asked[-1][0] is None and len(p.launched) == 2
    run2 = p.launched[1]["run"]
    assert TP.flags_of(p.launched[1]["cmd"])["--latent-body-output"] == str(retry_output)
    rec = record(p)
    assert rec["continuation"]["launched_at"] and rec["continuation"]["run"] == str(Path(run2).resolve())
    assert [a["run"] for a in rec["continuation"]["attempts"]] == [str(Path(run1).resolve()), str(Path(run2).resolve())]
    assert not TP.continuation_available(rec["continuation"])


def test_a_trainer_that_trained_before_it_died_used_the_continuation_and_no_rewrite_reopens_it(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "trains_then_exits")
    with pytest.raises(SystemExit) as exc:
        run_main(module, monkeypatch, "--tag", "x", "--reason", "the continuation")
    assert str(exc.value).startswith("relaunched trainer exited early with 0") and "stays unused" not in str(exc.value)
    retry_output = tmp_path / "latent" / "continuation_retry"
    with pytest.raises(TP.PreflightError, match="was used"):
        TP.set_continuation(p.fp, _retry_block(p, retry_output), "tester", root=p.root, registry=p.reg)
    p.mode["trainer"] = "stats"
    with pytest.raises(SystemExit, match="refused by the training preflight"):
        run_main(module, monkeypatch, "--tag", "z", "--reason", "a second continuation", "--set", "latent-body-output",
                 str(retry_output))
    assert len(p.launched) == 1 and "3 of 7 gates block launch" in p.asked[-1][0]


def test_an_exit_after_training_past_the_start_marks_the_continuation_used(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "trains_then_exits")
    with pytest.raises(SystemExit, match="exited early with 0"):
        run_main(module, monkeypatch, "--tag", "x", "--reason", "the continuation")
    rec = record(p)
    assert rec["continuation"]["launched_at"] and not TP.continuation_available(rec["continuation"])


def test_f7_the_module_texts_say_what_the_code_does():
    # the review of Oct 11, 2026, finding F7: normal training after a continuation is judged as before (allowed once
    # every gate reads PASS or justified N/A and a state follows the repair run: test_training_preflight.py
    # test_normal_training_after_the_continuation_is_judged_exactly_as_without_it), and records are one per fingerprint
    relaunch_text = " ".join(_module("relaunch_continuation_text").__doc__.split())
    preflight_text = " ".join(TP.__doc__.split())
    assert "normal training after it, are refused" not in relaunch_text
    assert "judged as normal training, which the record's FAIL gates refuse today" in relaunch_text
    assert "another continuation needs its own record" not in preflight_text
    assert "A configuration has one record (one per fingerprint), so a configuration takes at most one continuation" \
        in preflight_text
    assert "continues that run's end checkpoint" not in preflight_text                       # finding F4
    assert "is exactly that run's end checkpoint" in preflight_text


def test_launch_command_and_control_name_the_continuation(tmp_path, monkeypatch):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    module = _module("relaunch_continuation_units")
    monkeypatch.setattr(module.L, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(module.L, "stamp", lambda: "Oct 11, 2026, 9:00:00 AM PDT")
    preflight = SimpleNamespace(launch_plan=lambda cmd: TP.launch_plan(cmd, root=root, src=src))
    run = module.run_folder("x")
    cmd, changes, plan = module.launch_command(cont, end, run, [], preflight)
    assert plan["kind"] == "continuation" and TP.flags_of(cmd)["--max-steps"] == "4000"
    assert TP.launch_refusal(cmd, root=root, src=src, registry=reg) is None
    # a --set of a later end is kept and refused by the preflight; one of a shorter end past the first reading is judged
    cmd, changes, plan = module.launch_command(cont, end, run, [("max-steps", "4001")], preflight)
    assert "not bounded as declared" in TP.launch_refusal(cmd, root=root, src=src, registry=reg)
    cmd, changes, plan = module.launch_command(cont, end, run, [("max-steps", "3500")], preflight)
    assert TP.launch_refusal(cmd, root=root, src=src, registry=reg) is None
    # from another checkpoint the launch is normal training, unbounded, and refused while gates fail
    cmd, changes, plan = module.launch_command(cont, start, run, [], preflight)
    assert plan["kind"] == "normal" and TP.flags_of(cmd)["--max-steps"] == "0"
    assert "3 of 7 gates block launch" in TP.launch_refusal(cmd, root=root, src=src, registry=reg)
    found = {"cmd": "pause", "reason": "held", "previous_reason": "older", "archiver": "off"}
    ctl = module.control_after_launch(dict(kind="continuation", block="continuation", fingerprint=fp), "x", run, "4000",
                                      previous=found)
    assert ctl["cmd"] == "pause" and ctl["repair_run"] == str(run) and ctl["archiver"] == "off"
    assert "previous_reason" not in ctl and "authorizes no normal training" in ctl["reason"]
