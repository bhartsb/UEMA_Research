# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Re-check probes for commit e1aef5aea (the relaunch review's findings F1 to F12 fixed).

Written by the independent re-check of Oct 5, 2026 (docs/gain_of_function_v4/coordinator/
relaunch_repair_run_recheck_2026-10-05.md, branch review/relaunch-repair-run-fixes-2026-10-05, commit 6bbfb9fa1) and
taken into the suite by the coordinator with the fixes of its findings N1 to N6: the probes that recorded a failure as
it stood now assert the repaired behavior, its three strict-xfail probes pass and lost their marks, and three tests
cover the mutations no test caught (R4, R6, H3). The real hourly watcher runs on temporary folders with a simulated
clock; the relaunch script's main() runs with every process call replaced (the harness of
tests/test_relaunch_repair_run_review.py). Nothing here looks at, ends or starts a real trainer, supervisor or watcher,
and nothing is written outside pytest's temporary folder.
"""
import collections
import importlib.util
import json
import runpy
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tests.test_h84_hourly_watch import healthy_resources, row
from mimo_bridge.tests.test_relaunch_repair_run_review import SCRIPT, main_under_probe, record
from mimo_bridge.tests.test_training_preflight import STOP_CRITERIA, _bounded, _repair_setup
from mimo_bridge.tools import training_preflight as TP

REPO = Path(__file__).resolve().parents[4]
WATCHER = REPO / "docs/h84_launch_2026-09-18/hourly_watch.py"
T0 = 1_000_000.0
# Seconds from the trainer's start to its full stats records in production segment
# H84_fresh_drivers_20260918_seg41c_0845 (elapsed_s of the records at model steps 447,400, 447,500, 447,600 and
# 447,700, read Oct 5, 2026): one record per 100 model steps, then about 230 s apart.
SEG41C_ARRIVALS = (241, 467, 696, 934)
# The field a stop criterion names here: contact_control.gate_self_mean. In the last 14 seg41 segments it is absent
# from the first 2 full stats records of every segment and from the first 3 in seg41c_0845, seg41f_0322, seg41h_0223
# and seg41m_0402; it is in the newest record of seg41o_1431, the run of checkpoint 481,800 (read Oct 5, 2026).
CONTACT = dict(name="contact_gate_self", path="contact_control.gate_self_mean", op=">", line=0.9, from_run_hour=0,
               mode="stop", note="the self-contact gate's mean never above 0.9")


def arrivals(first=SEG41C_ARRIVALS, until=7200, every=230):
    times = list(first)
    while times[-1] + every <= until:
        times.append(times[-1] + every)
    return times


def watcher_started_before_the_first_record(tmp_path, monkeypatch, records, criteria=None, raw_criteria=None,
                                            hours=2, disk=None, patch=None):
    """The real hourly watcher as the supervisor starts it for a repair run since e1aef5aea: at the trainer's start,
    before the trainer's first full stats record. `records`: (seconds after the watcher started, record), appended to
    the stats file when the watcher's simulated clock passes them. Returns (the control.json it wrote or None, the
    second after its start at which it wrote its stop or None, the stop's reason or None). `patch`: {name: value} put
    into the watcher's globals before its main() runs (for example quick_breach, to judge the hourly sweep alone)."""
    healthy_resources(monkeypatch)
    if disk is not None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: disk)
    monkeypatch.setenv("UEMA_WATCH_SAVED", str(tmp_path / "_saved"))
    monkeypatch.setenv("UEMA_WATCH_ARCHIVE", str(tmp_path / "_archive"))
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "args.json").write_text("{}", encoding="utf-8")
    if raw_criteria is not None:
        (run / "declared_criteria.json").write_bytes(raw_criteria)
    else:
        (run / "declared_criteria.json").write_text(json.dumps(criteria), encoding="utf-8")
    stats = run / "replay_training_stats.jsonl"
    clock = {"t": 0.0}
    pending = sorted(records, key=lambda x: x[0])

    def advance(seconds):
        clock["t"] += seconds
        while pending and pending[0][0] <= clock["t"]:
            at, rec = pending.pop(0)
            with stats.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(rec, timestamp=T0 + at, elapsed_s=at)) + "\n")
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(run), str(hours)])
    monkeypatch.setattr(time, "sleep", advance)
    monkeypatch.setattr(time, "time", lambda: T0 + clock["t"])
    if patch:
        ns = runpy.run_path(str(WATCHER))                  # the watcher's functions; its main() not yet run
        ns["main"].__globals__.update(patch)               # run_path returns a copy of the globals
        ns["main"](sys.argv)
    else:
        runpy.run_path(str(WATCHER), run_name="__main__")
    ctl, reason = run / "control.json", run / "stop_reason.json"
    stop = json.loads(reason.read_text(encoding="utf-8")) if reason.exists() else None
    return ((json.loads(ctl.read_text()) if ctl.exists() else None),
            (stop["timestamp"] - T0) if stop else None, stop["reason"] if stop else None)


def contact_records(first_with_field=4):
    """Records at seg41c_0845's measured times; contact_control.gate_self_mean (0.2, under the 0.9 line) from the
    `first_with_field`-th record on, as production writes it."""
    out = []
    for i, at in enumerate(arrivals()):
        rec = row(step=447400 + 100 * i)
        if i + 1 >= first_with_field:
            rec["contact_control"] = dict(gate_self_mean=0.2, gate_self_line=0.5)
        out.append((at, rec))
    return out


def start_run_carries(a, **fields):
    """The start checkpoint's run's full stats records (its first 167 s after its start, then 100 model steps per
    273 s), the newest carrying `fields` (as seg41o_1431's does)."""
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01), cmd_response_cos_held=0.5,
        **fields)) + "\n" for step, elapsed in ((900, 167.0), (1000, 440.0))), encoding="utf-8")


# ------------------------------------------------------------------ N1: a stop field absent from the first records
def test_a_stop_field_written_from_the_fourth_report_and_holding_its_line_stops_nothing(tmp_path, monkeypatch):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    start_run_carries(a, contact_control=dict(gate_self_mean=0.2))
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    assert TP.repair_block_problems(dict(block, stop_criteria=[CONTACT]), reg) == []     # the preflight accepts it
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "w", monkeypatch, contact_records(),
                                                            criteria=[CONTACT])
    # before the fix the first check stopped the run at 900 s on "contact_control.gate_self_mean is absent"
    assert stop is None, why


def test_a_stop_field_present_from_the_first_report_stops_nothing(tmp_path, monkeypatch):
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, contact_records(1),
                                                            criteria=[CONTACT])
    assert stop is None, why


def test_a_stop_field_absent_for_an_hour_of_records_stops_the_run(tmp_path, monkeypatch):
    # a misspelled path (finding F3) still stops the run, once its records span an hour after the line's start
    typo = [dict(CONTACT, path="contact_control.gate_self_maen")]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, contact_records(1), criteria=typo)
    assert stop == {"cmd": "snapshot_and_stop"} and "gate_self_maen is absent" in why
    first_record_at = SEG41C_ARRIVALS[0]
    assert at >= first_record_at + 3600 and at <= first_record_at + 3600 + 600


# ------------------------------------------------------------------ F4 and N2: when the first declared stop is judged
def test_a_watcher_started_before_the_first_record_judges_a_declared_stop_300_seconds_after_that_record(
        tmp_path, monkeypatch):
    # a repair run bounded at start + 300 model steps: its records come at 241, 467 and 696 s; the vision causal error
    # reads 0.1 against the stop line of 0.04 from the first record on. Before the fix the first judgement came at
    # 900 s, after the run's last record; the watcher now looks for the first record every 30 s (found at 270 s)
    records = [(at, row(step=1100 + 100 * i, causal_pe_per_modality=dict(vision=0.1, tactile=.04,
                                                                         proprioception=.005)))
               for i, at in enumerate(SEG41C_ARRIVALS[:3])]
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "w", monkeypatch, records,
                                                            criteria=STOP_CRITERIA)
    assert stop == {"cmd": "snapshot_and_stop"} and at == 270 + 300 and "5-minute check" in why
    assert at <= SEG41C_ARRIVALS[2]                     # judged while the run trains


def test_a_bound_before_the_first_reading_is_refused_at_launch(tmp_path):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    with pytest.raises(TP.PreflightError, match="before the watcher first judges stop line"):
        TP.set_repair_run(fp, dict(block, max_model_steps=150), "tester", root=root, registry=reg)
    rec = TP._load(TP.record_path(fp, root))
    rec["repair_run"]["max_model_steps"] = 150                           # a block edited by hand after it was written
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(recorded, 1150), root=root, src=src, registry=reg)
    assert why and "before the watcher first judges stop line" in why


# ------------------------------------------------------------------ N4: other ways the stricter reading stops a run
def test_a_declared_criteria_file_with_a_byte_order_mark_is_read(tmp_path, monkeypatch):
    # PowerShell 5.1's Out-File and Set-Content -Encoding utf8 write a byte-order mark (the style guide, control.json)
    raw = b"\xef\xbb\xbf" + json.dumps(STOP_CRITERIA).encode("utf-8")
    records = [(at, row(step=1100 + 100 * i)) for i, at in enumerate(SEG41C_ARRIVALS)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, records, raw_criteria=raw)
    assert stop is None, why


def test_a_stop_entry_without_a_path_stops_the_run_naming_it_and_never_skips_the_disk_line(tmp_path, monkeypatch):
    low_disk = collections.namedtuple("usage", "total used free")(1e12, 0.0, 5e9)        # 5 GB, under the 8 GB line
    records = [(at, row(step=1100 + 100 * i)) for i, at in enumerate(SEG41C_ARRIVALS)]
    bad = [dict(name="hand_written", op=">", line=1, mode="stop")]                       # no path
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "disk", monkeypatch, records, criteria=bad,
                                                            disk=low_disk)
    assert stop == {"cmd": "snapshot_and_stop"} and "free disk" in why                 # the disk line comes first
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "bad", monkeypatch, records, criteria=bad)
    assert stop == {"cmd": "snapshot_and_stop"} and "declared stop criterion 1 cannot be judged" in why


# ------------------------------------------------------------------ H3: the hourly sweep's absent-field stop
def test_the_hourly_sweep_stops_on_a_stop_field_absent_for_an_hour_of_records(tmp_path):
    ns = runpy.run_path(str(WATCHER))
    run = tmp_path / "run"
    run.mkdir()
    (run / "declared_criteria.json").write_text(json.dumps([dict(CONTACT, path="contact_control.gate_self_maen")]),
                                                encoding="utf-8")
    ns["declared_criteria"].__globals__["RUN"] = str(run)        # run_path returns a copy of the globals
    window = [dict(type="mimo", step=1000, contact_control=dict(gate_self_mean=0.2))]
    assert "is absent" in ns["declared_criteria"](window, 1.5, 3800.0)
    assert ns["declared_criteria"](window, 1.5, 1800.0) is None               # an hour of records not yet passed
    assert ns["declared_criteria"](window, 0.5, 3800.0) is None               # run-hour under the line's start + 1


# ------------------------------------------------------------------ F9 and N3: the state added after the launch
def test_a_state_added_after_the_launch_must_continue_the_repaired_model(tmp_path):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    (tmp_path / "experiments/run_repair").mkdir(parents=True)        # marking needs the run folder (round two, R5)
    TP.mark_launched(fp, str(tmp_path / "experiments/run_repair"), "tester", root=root)
    time.sleep(0.05)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "PASS", "measured on the repair run's end checkpoint", ["row_%db" % n], "tester",
                    root=root)
    assert "ran its repair run" in TP.launch_refusal(recorded, root=root, src=src, registry=reg)
    # the start checkpoint, whose weights the repair run never changed, is refused as a state after the launch
    with pytest.raises(TP.PreflightError, match="is not one"):
        TP.add_state(fp, str(start), "gates 4, 5 and 7 judged again", "tester", root=root)
    # and so is a checkpoint of a run that never saw the repair run
    other = tmp_path / "experiments/run_other/checkpoints/ckpt_step_5000"
    other.mkdir(parents=True)
    (other.parents[1] / "args.json").write_text(json.dumps({"resume": None}), encoding="utf-8")
    with pytest.raises(TP.PreflightError, match="is not one"):
        TP.add_state(fp, str(other), "gates 4, 5 and 7 judged again", "tester", root=root)
    assert TP.launch_refusal(recorded, root=root, src=src, registry=reg)


def test_a_state_written_by_hand_that_does_not_continue_the_repair_run_authorizes_nothing(tmp_path):
    # the record edited by hand past add_state's refusal: the launch reads only states that continue the repair run
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    (tmp_path / "experiments/run_repair").mkdir(parents=True)        # marking needs the run folder (round two, R5)
    TP.mark_launched(fp, str(tmp_path / "experiments/run_repair"), "tester", root=root)
    time.sleep(0.05)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "PASS", "measured", ["row_%db" % n], "tester", root=root)
    rec = TP._load(TP.record_path(fp, root))
    rec["states"].append(dict(checkpoint=str(start), at="later", at_s=time.time() + 1, by="hand", reason="hand"))
    TP._save(TP.record_path(fp, root), rec)
    assert TP.launch_refusal(recorded, root=root, src=src, registry=reg)


# ------------------------------------------------------------------ N5 and N6: the supervisor attached at the start
def test_a_supervisor_that_ends_at_once_names_the_trainers_exit(tmp_path, monkeypatch):
    # supervise() raises "Attach requires exactly one live trainer" when the trainer died before it looked; psutil then
    # finds no supervisor process (measured on this machine: psutil.Process of an exited child raises NoSuchProcess)
    import psutil
    module, p = main_under_probe(tmp_path, monkeypatch, "exits")

    class Gone:
        def __init__(self, pid):
            raise psutil.NoSuchProcess(pid)

    def popen(cmd, **kw):
        if kw.get("stdout") is not None:
            kw["stdout"].close()
        p.started.append(dict(cmd=[str(c) for c in cmd]))
        return SimpleNamespace(pid=777)                                  # it wrote no state before it ended
    monkeypatch.setattr(psutil, "Process", Gone)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(SystemExit, match="ended before it reported the run") as caught:
        module.main()
    assert "had ended with exit code 1" in str(caught.value)
    run = p.launched["run"]
    assert json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
    assert not record(p)["launched_at"]


def test_after_an_early_death_a_relaunch_without_parent_names_the_run_to_pass(tmp_path, monkeypatch):
    # the real supervisor reports the run at its first sweep while the trainer still loads; the trainer then dies
    module, p = main_under_probe(tmp_path, monkeypatch, "exits")
    with pytest.raises(SystemExit, match="exited early with 1"):
        module.main()
    run = p.launched["run"]
    state = json.loads((p.saved / "training_continuation.json").read_text(encoding="utf-8"))
    assert state["active_run"] == str(run)               # the relaunch's default parent is now the dead repair run
    spec = importlib.util.spec_from_file_location("relaunch_recheck_parent", SCRIPT)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    monkeypatch.setattr(fresh.L, "RECEIPT", tmp_path / "receipt.json")
    with pytest.raises(SystemExit, match="has no complete eight-world checkpoint") as caught:
        fresh.newest_complete_checkpoint(Path(state["active_run"]))
    assert "pass --parent %s" % p.a in str(caught.value)


def test_an_exception_between_the_trainers_start_and_the_attach_stops_the_repair_trainer(tmp_path, monkeypatch):
    # finding N6: L.note(event="relaunch: trainer launched") ran after L.launch_trainer and before the protected attach
    module, p = main_under_probe(tmp_path, monkeypatch, "silent")
    note0 = module.L.note

    def note(**kw):
        if kw.get("event") == "relaunch: trainer launched":
            raise OSError("the launch receipt could not be written")
        return note0(**kw)
    monkeypatch.setattr(module.L, "note", note)
    with pytest.raises(SystemExit, match="OSError: the launch receipt could not be written"):
        module.main()
    run = p.launched["run"]
    assert p.launched and not p.started                     # the trainer started; no supervisor
    assert json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"


def test_a_watcher_started_before_the_trainer_writes_args_json_waits_for_it(tmp_path, monkeypatch):
    # the supervisor's first sweep now comes seconds after the trainer starts; train_mimo.main writes args.json after
    # its imports
    run = tmp_path / "run"
    run.mkdir()
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        if len(slept) == 3:
            (run / "args.json").write_text("{}", encoding="utf-8")          # the trainer writes it 45 s in
        if len(slept) > 3:
            raise KeyboardInterrupt                                           # configured: end the probe here
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(run), "1"])
    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        runpy.run_path(str(WATCHER), run_name="__main__")
    assert slept[:3] == [15, 15, 15]
    # a run folder that never gets args.json ends the watcher after 600 s of looking
    other = tmp_path / "other"
    other.mkdir()
    waited = []
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(other), "1"])
    monkeypatch.setattr(time, "sleep", lambda s: waited.append(s))
    with pytest.raises(FileNotFoundError):
        runpy.run_path(str(WATCHER), run_name="__main__")
    assert sum(waited) == 600


# ------------------------------------------------------------------ R4 and R6: the relaunch's own branches
def test_a_segment_row_write_that_fails_otherwise_than_refused_leaves_the_script_running(tmp_path, monkeypatch):
    # mutation R4: the catch narrowed back to test_runner.Refused ended the script before its end
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")
    monkeypatch.setattr(module, "segment_row_refusal", lambda row_id, root=None: None)
    from mimo_bridge.tools import test_runner

    def segment_start(*a, **k):
        raise RuntimeError("the registry is locked")
    monkeypatch.setattr(test_runner, "segment_start", segment_start)
    monkeypatch.setattr(sys, "argv", ["relaunch.py", "--tag", "x", "--reason", "review probe", "--row", "the_row"])
    module.main()
    events = [e["event"] for e in json.loads((p.saved / "seg41_launch_receipt.json").read_text())["events"]]
    assert "relaunch: SEGMENT ROW NOT WRITTEN" in events and record(p)["launched_at"]


def test_trained_past_reads_stage_done_when_no_stats_record_exists(tmp_path):
    # mutation R6: trained_past ignored stage_done.json
    spec = importlib.util.spec_from_file_location("relaunch_recheck_trained", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    run.mkdir()
    assert module.trained_past(run, 1000) is False
    (run / "stage_done.json").write_text(json.dumps({"steps": 1000}), encoding="utf-8")
    assert module.trained_past(run, 1000) is False                       # it ended at its start
    (run / "stage_done.json").write_text(json.dumps({"steps": 1050}), encoding="utf-8")
    assert module.trained_past(run, 1000) is True


# ------------------------------------------------------------------ F2: the pause watchdog's test of a live trainer
def test_trainer_of_matches_the_forms_the_relaunch_writes_and_only_the_run_folder_itself(monkeypatch):
    from mimo_bridge.tools import pause_watchdog as PW
    run = r"C:\Users\x\Saved\mimo_bridge\experiments\H84_fresh_drivers_20260918_seg41r_1700"

    def processes(*cmdlines):
        return lambda *a, **k: [SimpleNamespace(info={"cmdline": c}) for c in cmdlines]
    trainer = [r"C:\venv\Scripts\python.exe", "-u", "-m", "mimo_bridge.train_mimo", "--out", run,
               "--checkpoint-storage", run + r"\checkpoints", "--max-steps", "482100"]
    monkeypatch.setattr(PW.psutil, "process_iter", processes(trainer))
    for form in (run, run.replace("\\", "/"), run.upper(), run + "\\"):
        assert PW.trainer_of(form), form
    assert not PW.trainer_of(r"\\?" + "\\" + run)          # the extended-length form, which the relaunch never writes
    assert not PW.trainer_of(run + "_other")
    job = ["python", "-m", "mimo_bridge.train_mimo", "--resume", run + r"\checkpoints\ckpt_step_481900", "--out",
           r"D:\eval\x"]
    monkeypatch.setattr(PW.psutil, "process_iter", processes(job))
    assert not PW.trainer_of(run)                            # a job resuming the repair run's checkpoint
