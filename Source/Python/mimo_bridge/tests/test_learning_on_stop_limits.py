# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/learning_on_stop_limits/: the learning-on stop-limit measurement on stubs and synthetic
records (no learner, no checkpoint, no saved world, no graphics card).

  THE ARM - the six values replaced (folders inside the arm's own folder, --resume its own copy, --max-steps 481,800 plus
  the declared steps), every other value the command file's; the two command files differ in the package's three
  settings only.
  THE LIMITS - the rule on synthetic control runs: from the window the arms measure (model steps 481,900 to 482,600)
  alone, s_D as the root mean square of C1 - C2, rho counted from 0, n_eff, k the t quantile at 1 - 0.05 / 4 on the
  window's degrees of freedom, the session check at 1 - 0.01 / 8, the two-sided band at 0.975; the whole run beside;
  the gross line training's highest record plus 5 x the whole run's s_D; vestibular beside, never a limit.
  THE EMERGENCY STOPS - each rule on synthetic records, events, machine and card readings; the stats stream read in
  complete lines; a crossing in the record written as the arm ends read under the same rules.
  THE JOB - the arms one at a time on stub children: complete arms compared; a reading stop of the finger arm read as a
  finding and the next arm run; a reading stop of a control arm, a machine stop (the commit charge, the arm's deadline,
  a foreign python process or full memory on the graphics card, another trainer) or an arm that ends in an error ending
  the job; the refusals before any copy (the length, the arms, a checkpoint other than 481,800, another trainer code,
  command file or limits file); the original checkpoint hashed before and after; the learned states discarded and the
  records kept.
  THE COMPARISON - the N-record mean difference against the declared limit; the session's control pair; the arms'
  settings; the package's replacement event; a development run passing on its plumbing alone, its session pair a
  finding; what each verdict sets; beside, deciding nothing: each arm's distance below the gross lines, vestibular's
  difference, each sense's rise and band beside the stop list's line.
  THE SECOND RE-CHECK OF OCT 9 - the trainer rules on python processes only; the settings through the listing's own
  psutil objects, a reused id skipped, the processes below normal after the first poll counted; the pause read as the
  supervisor reads it; the job's deadline in the wait and before each copy; a process left alive after the end wait
  holding the next arm; a development run's gof; the job's own priority; a listed tree with children; a real process
  tree and real junctions (on C:), only processes and folders these tests make.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
TOOL_DIR = REPO / "docs/gain_of_function_v4/learning_on_stop_limits"
SENSES = ("tactile", "proprioception", "vision", "audio")


@pytest.fixture(scope="module")
def lo():
    sys.path.insert(0, str(TOOL_DIR))
    try:
        return importlib.import_module("learning_on")
    finally:
        pass


@pytest.fixture(scope="module")
def arm():
    sys.path.insert(0, str(TOOL_DIR))
    return importlib.import_module("learning_on_arm")


def stats_record(step, ratios, joint=14.3, nan=0, elapsed=None, ts=None, transitions=700, replay_loss=0.09,
                 replay_steps=50):
    """A full stats record of the trainer's form (the fields the measurement reads)."""
    k = (int(step) - 481800) / 100.0
    return dict(type="mimo", step=int(step), nan_steps=nan, elapsed_s=(250.0 * k) if elapsed is None else elapsed,
                timestamp=(1000.0 + 250.0 * k) if ts is None else ts, hand_control=dict(joint_error_deg=joint),
                replay_loss=replay_loss, next_frame_norm=dict(replay_steps=replay_steps),
                prediction_audit_per_body=dict(senses={s: dict(without_body_2=dict(ratio=ratios[s] / 1000.0,
                                                                                   transitions=transitions))
                                                       for s in ratios}))


BASE = dict(tactile=940.0, proprioception=690.0, vision=650.0, audio=900.0, vestibular=850.0)


def control_runs(n=28, spread=(10.0, 60.0, 40.0, 30.0, 20.0), later=1.0):
    """Synthetic C1, C2 and L records from 481,900: C1 - C2 alternates +/- spread per sense, vestibular last (s_D =
    spread, rho below 0, counted as 0); later multiplies the spread of the records after the window the arms measure
    (model steps 481,900 to 482,800)."""
    c1, c2, ll = {}, {}, {}
    names = SENSES + ("vestibular",)
    for i in range(n):
        step = 481900 + 100 * i
        sign = 1 if i % 2 == 0 else -1
        f = 1.0 if step <= 482800 else later
        r1 = {s: BASE[s] + sign * f * d / 2.0 for s, d in zip(names, spread)}
        r2 = {s: BASE[s] - sign * f * d / 2.0 for s, d in zip(names, spread)}
        c1[step], c2[step] = stats_record(step, r1, joint=14.3 + 0.25 * sign), stats_record(step, r2, joint=14.3 - 0.25 * sign)
        ll[step] = stats_record(step, dict(BASE))
    return c1, c2, ll


# ------------------------------------------------------------------------------------------------ the arm
def test_the_arm_replaces_exactly_the_six_values_and_runs_the_declared_length(arm, tmp_path):
    U = arm.U
    scratch = tmp_path / "job" / "arm_control"
    copy = scratch / "checkpoint_copy" / "ckpt_step_481800"
    values = arm.arm_values(scratch, copy, 1000)
    assert values["--max-steps"] == 482800 and sorted(values) == sorted(U.REPLACED)
    assert arm.replacement_problems(values, scratch, copy, 1000) == []
    argv = U.train_argv(json.loads(arm.ARM_COMMANDS["control"].read_text(encoding="utf-8")))
    new, changes = U.replace_arguments(argv, values)
    assert len(U.arguments_differ(argv, new)) == 6 and {c["flag"] for c in changes} == set(U.REPLACED)
    # a copy outside the arm's folder, a folder under Saved/, another length: refused
    outside = dict(values, **{"--resume": tmp_path / "elsewhere" / "ckpt"})
    assert any("not the arm's checkpoint copy" in p for p in arm.replacement_problems(outside, scratch, copy, 1000))
    saved = dict(values, **{"--out": tmp_path / "Saved" / "run"})
    assert any("Saved/" in p for p in arm.replacement_problems(saved, scratch, copy, 1000))
    assert any("--max-steps" in p for p in arm.replacement_problems(values, scratch, copy, 700))


def test_the_two_command_files_differ_in_the_package_settings_only(arm):
    U = arm.U
    control = U.train_argv(json.loads(arm.ARM_COMMANDS["control"].read_text(encoding="utf-8")))
    finger = U.train_argv(json.loads(arm.ARM_COMMANDS["finger"].read_text(encoding="utf-8")))

    def flags(argv):
        out, i = {}, 0
        while i < len(argv):
            if argv[i].startswith("--"):
                has = i + 1 < len(argv) and not argv[i + 1].startswith("--")
                out[argv[i]] = argv[i + 1] if has else True
                i += 2 if has else 1
            else:
                i += 1
        return out
    a, b = flags(control), flags(finger)
    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == sorted(arm.PACKAGE_FLAGS)
    assert arm.ARM_COMMANDS["control_again"] == arm.ARM_COMMANDS["control"]


# ------------------------------------------------------------------------------------------------ the limits
def test_the_limits_follow_the_declared_rule(lo):
    from scipy.stats import t as student_t
    # the records after the window differ three times as much: the declared limits come from the window alone
    c1, c2, ll = control_runs(later=3.0)
    limits = lo.compute_limits(dict(C1=c1, C2=c2), dict(C1=c1, C2=c2, L=ll))
    assert limits["window"] == dict(first=481900, last=482800, n=10)
    t = limits["senses"]["tactile"]
    assert t["n"] == 10 and abs(t["s_D"] - 10.0) < 1e-9 and t["rho"] < 0 and t["df"] == 10
    k, se10 = float(student_t.ppf(1 - 0.05 / 4, 10)), 10.0 / math.sqrt(10)
    assert abs(t["k"] - k) < 1e-12 and abs(t["by_records"]["10"]["limit"] - k * se10) < 1e-9
    assert abs(t["k_check"] - float(student_t.ppf(1 - 0.01 / 8, 10))) < 1e-12
    assert abs(t["by_records"]["10"]["check"] - float(student_t.ppf(1 - 0.01 / 8, 10)) * se10) < 1e-9
    assert abs(t["by_records"]["10"]["rise_found_95"] - (k + float(student_t.ppf(0.95, 10))) * se10) < 1e-9
    assert abs(t["by_records"]["10"]["band_95"] - float(student_t.ppf(0.975, 10)) * se10) < 1e-9
    assert sorted(t["by_records"], key=int) == [str(N) for N in range(1, 11)]         # the window's lengths only
    # the whole run beside, deciding nothing: here its spread is the larger one
    whole = limits["whole_run"]["tactile"]
    assert whole["n"] == 28 and whole["s_D"] > 20.0 and whole["by_records"]["10"]["limit"] > t["by_records"]["10"]["limit"]
    # the gross line: training's highest record of the three runs plus 5 x the whole run's s_D (the accepted lines)
    assert abs(limits["gross"]["tactile"]["highest_record"] - 955.0) < 1e-9
    assert abs(limits["gross"]["tactile"]["line"] - (955.0 + 5 * whole["s_D"])) < 1e-9
    # vestibular beside, never among the limits
    assert "vestibular" not in limits["senses"] and abs(limits["beside"]["vestibular"]["window"]["s_D"] - 20.0) < 1e-9
    # the finger joint error's band from the window, the whole run beside
    assert abs(limits["joint"]["s_D"] - 0.5) < 1e-9 and limits["joint"]["whole_run"]["s_D"] > 0
    # a positive lag-1 correlation widens the limit beyond the square root of the records
    rising = {int(k): v for k, v in json.loads(json.dumps(ll)).items()}
    for i, step in enumerate(sorted(rising)):
        rising[step]["prediction_audit_per_body"]["senses"]["tactile"]["without_body_2"]["ratio"] = \
            (BASE["tactile"] + 10.0 * math.sin(i / 3.0)) / 1000.0
    wide = lo.compute_limits(dict(C1=rising, C2=ll), dict(C1=rising, C2=ll, L=ll))["senses"]["tactile"]
    assert wide["rho"] > 0 and wide["by_records"]["8"]["n_eff"] < 8


def test_the_limits_read_only_the_manifests_copies(lo, tmp_path):
    copies = tmp_path / "records"
    (copies / "C1" / "run_files").mkdir(parents=True)
    stats = copies / "C1" / "run_files" / "replay_training_stats.jsonl"
    stats.write_text(json.dumps(stats_record(481900, BASE)) + "\n", encoding="utf-8")
    manifest = dict(runs=dict(C1=dict(rows=[dict(copy=str(stats), copy_sha256=lo.sha256_file(stats))])))
    (copies / "manifest_run_files.json").write_text(json.dumps(manifest), encoding="utf-8")
    recs, info = lo.copy_records("C1", copies=copies)
    assert list(recs) == [481900] and info["sha256"] == lo.sha256_file(stats)
    stats.write_text(json.dumps(stats_record(481900, dict(BASE, tactile=999.0))) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        lo.copy_records("C1", copies=copies)


# ------------------------------------------------------------------------------------------------ the emergency stops
def test_each_emergency_rule_reads_its_own_record(lo):
    c1, c2, ll = control_runs()
    limits = lo.compute_limits(dict(C1=c1, C2=c2), dict(C1=c1, C2=c2, L=ll))
    good = stats_record(481900, BASE)
    assert lo.record_stop(good, limits) is None
    assert "non_finite_update" in lo.record_stop(stats_record(481900, BASE, nan=2), limits)
    missing = stats_record(481900, {s: v for s, v in BASE.items() if s != "audio"})
    assert "no finite audio prediction ratio" in lo.record_stop(missing, limits)
    gross = stats_record(481900, dict(BASE, proprioception=limits["gross"]["proprioception"]["line"] + 1.0))
    assert "above its gross line" in lo.record_stop(gross, limits)
    no_joint = stats_record(481900, BASE)
    no_joint["hand_control"] = {}
    assert "finger joint error" in lo.record_stop(no_joint, limits)
    # the stop list's finger joint error entry: the hour mean above 15.0 degrees, judged from run-hour 0.5
    early = [stats_record(481900 + 100 * i, BASE, joint=16.0, elapsed=250.0 * (i + 1)) for i in range(5)]
    assert lo.joint_hour_stop(early) is None                    # 1,250 s: before run-hour 0.5
    late = [stats_record(481900 + 100 * i, BASE, joint=16.0, elapsed=250.0 * (i + 1), ts=250.0 * (i + 1))
            for i in range(8)]
    assert "hour mean 16.00" in lo.joint_hour_stop(late)
    fine = [stats_record(481900 + 100 * i, BASE, joint=14.9, elapsed=250.0 * (i + 1), ts=250.0 * (i + 1))
            for i in range(8)]
    assert lo.joint_hour_stop(fine) is None
    assert "physical fault" in lo.event_stop(dict(type="episode_reset", step=482010, reason="nan qpos"))
    assert lo.event_stop(dict(type="video", step=482010)) is None


def test_the_machine_and_card_stops(lo):
    assert lo.machine_stop((70.0, 84.4, 3.0)) is None
    assert "within 6 GB" in lo.machine_stop((78.5, 84.4, 1.0))
    assert "page-file drive" in lo.machine_stop((40.0, 84.4, 9.0), short=[("C:/", 11.0)])
    assert lo.machine_stop(None) is None
    card = dict(apps=[dict(pid=10, name="C:/x/python.exe")], memory_used_mib=[7074])
    assert lo.gpu_stop(card, [10, 11]) is None                  # the arm's own trainer
    assert "not the arm's" in lo.gpu_stop(card, [11])
    assert "at or above 8100" in lo.gpu_stop(dict(apps=[], memory_used_mib=[8100]), [1])
    assert lo.gpu_stop(dict(error="unreadable"), [1]) is None


def test_the_stats_stream_is_read_in_complete_lines(lo, tmp_path):
    path = tmp_path / "s.jsonl"
    reader = lo.StatsReader(path)
    assert reader.new() == []
    path.write_bytes(b'{"type": "launch", "step": 481800}\n{"type": "mi')
    got = reader.new()
    assert [r["type"] for r in got] == ["launch"]
    with open(path, "ab") as f:
        f.write(b'mo", "step": 481900}\n')
    assert [r["step"] for r in reader.new()] == [481900] and reader.new() == []


# ------------------------------------------------------------------------------------------------ the job on stub children
class _Arm:
    """A stub trainer arm: it writes the run's args.json at its start (as train_mimo does), each poll writes the next line
    of its stats stream; when the lines are spent it writes the arm record and returns its exit code. last_with_exit:
    the poll that writes the last line also ends the arm (a record written just before the trainer exits)."""

    def __init__(self, cmd, lines, code=0, args_extra=None, problems=(), last_with_exit=False, hang=False):
        self.cmd = cmd
        self.name = cmd[cmd.index("--arm") + 1]
        self.folder = Path(cmd[cmd.index("--scratch") + 1])
        self.record = Path(cmd[cmd.index("--out") + 1])
        self.lines, self.code, self.done, self.killed = list(lines), code, False, False
        self.args_extra, self.problems, self.last_with_exit = dict(args_extra or {}), list(problems), last_with_exit
        self.hang = hang                                         # never ends by itself: only a stop ends it
        self.pid = 70000 + len(self.name)
        (self.folder / "run").mkdir(parents=True, exist_ok=True)
        for sub in ("run/checkpoints", "caregiver_record/checkpoint_state", "latent_body"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        (self.folder / "run" / "checkpoints" / "model.safetensors").write_bytes(b"w" * 64)
        (self.folder / "caregiver_record" / "checkpoint_state" / "c.pt.gz").write_bytes(b"c" * 32)
        (self.folder / "latent_body" / "latest.pt").write_bytes(b"l" * 16)
        args = dict(out=str(self.folder / "run"), resume=str(self.folder / "checkpoint_copy"), seed=0,
                    max_steps=482800, **self.args_extra)
        (self.folder / "run" / "args.json").write_text(json.dumps(args), encoding="utf-8")

    def poll(self):
        if self.killed:
            return -9
        if self.lines:
            with open(self.folder / "run" / "replay_training_stats.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(self.lines.pop(0)) + "\n")
            if self.lines or not self.last_with_exit:
                return None
        if self.hang:
            # the job's clock advances 1 s per poll here, so the 1,010 s deadline of 100 model steps ends the arm
            # within about 1,011 polls; 5,000 polls mean the job never stopped it
            self.polls = getattr(self, "polls", 0) + 1
            if self.polls > 5000:
                raise AssertionError("the job never stopped an arm that does not end")
            return None
        if not self.done:
            self.done = True
            self.record.write_text(json.dumps(dict(arm=self.name, problems=self.problems)), encoding="utf-8")
        return self.code

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        """The job waits for a stopped arm (end_and_wait): the stub records the time limit it was given."""
        self.waited = timeout
        return -9 if self.killed else self.code


def _lines(name, steps=1000, rise=None, extra=None):
    out = [dict(type="launch", step=481800, resumed_from="copy")]
    if name == "finger":
        out.append(dict(type="hand_calibration_replacement", step=481800))
    for i in range(steps // 100):
        step = 481900 + 100 * i
        ratios = dict(BASE)
        for s, v in (rise or {}).items():
            ratios[s] += v
        out.append(stats_record(step, ratios, joint=13.2 if name == "finger" else 14.3))
        if extra and step in extra:
            out[-1].update(extra[step])
    return out


def _job(lo, tmp_path, plans, steps=1000, declared=1000, arms="control,finger,control_again", machine=None,
         expected_model=None, code_tree=None, gpu=None, during=None, limits_edit=None, preconditions_fn=None,
         priority=None, paused=None, end=None, listing=None, alive=None, own_priority=None, clock=None,
         deadline_s=0.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    c1, c2, ll = control_runs()
    limits = lo.compute_limits(dict(C1=c1, C2=c2), dict(C1=c1, C2=c2, L=ll))
    if limits_edit:
        limits_edit(limits)
    lpath = tmp_path / "limits.json"
    lpath.write_text(json.dumps(limits), encoding="utf-8")
    ck = tmp_path / "Ckpt" / "ckpt_step_481800"
    ck.mkdir(parents=True)
    for name, data in (("model.safetensors", b"m" * 64), ("optimizer.pt", b"o" * 64), ("runtime_state.pt", b"r" * 64)):
        (ck / name).write_bytes(data)
    comp = tmp_path / "Ckpt" / "companion.pt.gz"
    comp.write_bytes(b"companion")
    (ck / "episode_state.json").write_text(json.dumps(dict(storage_path=str(comp), sha256=lo.sha256_file(comp))),
                                          encoding="utf-8")
    made = []

    def popen(cmd, **kw):
        name = cmd[cmd.index("--arm") + 1]
        p = _Arm(cmd, **plans[name])
        made.append(p)
        return p
    # the job's clock: 1 s per reading; a test may pass its own dict to move it further
    clock = dict(t=0.0) if clock is None else clock

    def now():
        clock["t"] += 1.0
        return clock["t"]
    a = argparse.Namespace(scratch=str(tmp_path / "job"), out=str(tmp_path / "result.json"), steps=steps,
                           declared_steps=declared, arms=arms, limits=str(lpath), limits_sha256=lo.sha256_file(lpath),
                           deadline_s=deadline_s)
    # the stubs never touch a real process: the priority setting, the job's own priority, the pause and the
    # other-trainer scan are given
    code = lo.cmd_job(a, popen=popen, poll_s=0.0, contain=False, checkpoint=ck,
                      preconditions_fn=preconditions_fn or (lambda: []),
                      supervisor_env=dict(PATH="x"), machine=machine or (lambda: ((20.0, 84.4, 30.0), [])),
                      gpu=gpu or (lambda: dict(apps=[], memory_used_mib=[100])), now=now, sleep=lambda s: None,
                      during=during or (lambda own, root=None: []), code_tree=code_tree,
                      expected_model=(lo.sha256_file(ck / "model.safetensors") if expected_model is None
                                      else expected_model),
                      priority=priority or (lambda procs: {}), paused=paused or (lambda: (True, "cmd 'pause'")),
                      end=end, listing=listing, alive=alive,
                      own_priority=own_priority or (lambda: dict(priority_before=16384, priority=32)))
    res = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    job = json.loads((tmp_path / "job" / "job_record.json").read_text(encoding="utf-8"))
    return code, res, job, made, ck


def test_the_job_runs_the_arms_in_sequence_and_compares(lo, tmp_path):
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger", rise=dict(tactile=2.0))),
                 control_again=dict(lines=_lines("control_again"),
                                    args_extra=dict()))
    plans["finger"]["args_extra"] = dict(finger_joint_servo="on", hand_response_calibration="refit",
                                         replace_saved_hand_calibration="record")
    code, res, job, made, ck = _job(lo, tmp_path, plans)
    assert [p.name for p in made] == ["control", "finger", "control_again"]
    assert res["verdict"] == "pass", res["verdict_reason"]
    assert code == 0 and abs(res["stages"]["prediction_tactile"]["difference"] - 2.0) < 1e-9
    assert res["finger_joint_error"]["difference"] < 0
    # what a pass sets, and what stands beside the stop list's lines
    assert res["sets"].startswith("the matched comparison's stop list stays as declared")
    tl = res["beside_stop_list"]["tactile"]
    assert tl["stop_line_in_1000"] == 990.0 and abs(tl["rise"] - 2.0) < 1e-9 and tl["band_95"] > 0
    # each arm's distance below the gross lines: control's highest tactile record 940 against 945 + 5 x 10
    gd = res["gross_distance"]["control"]
    assert gd["control"] is True and abs(gd["senses"]["tactile"]["below_line"] - 55.0) < 1e-9
    assert res["gross_distance"]["finger"]["control"] is False
    # vestibular reported beside, not judged
    v = res["beside_senses"]["vestibular"]
    assert v["judged"] is False and abs(v["difference"]) < 1e-9 and abs(v["window_s_D"] - 20.0) < 1e-9
    assert "vestibular" not in str(res["stages"])
    # each arm its own fresh copy, the original read before and after and unchanged
    assert all((tmp_path / "job" / ("arm_%s" % n) / "checkpoint_copy").exists() is False for n in ("control", "finger"))
    assert job["original_changed"] == [] and job["original_before"]["files"] == job["original_after"]["files"]
    # the learned states discarded, the records kept
    gone = {Path(r["path"]).name for r in job["discarded"]["removed"]}
    assert {"checkpoint_copy", "checkpoints", "checkpoint_state", "latest.pt"} <= gone
    assert (tmp_path / "job" / "arm_finger" / "run" / "replay_training_stats.jsonl").is_file()


def test_a_finding_and_the_stops(lo, tmp_path):
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    # the finger arm's tactile ratio rises beyond its limit: a finding against the package
    plans = dict(control=dict(lines=_lines("control")),
                 finger=dict(lines=_lines("finger", rise=dict(tactile=30.0)), args_extra=pkg),
                 control_again=dict(lines=_lines("control_again", rise=dict(tactile=3.0))))
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans)
    assert res["verdict"] == "fail" and code == 1 and res["stages"]["prediction_tactile"]["met"] is False
    assert res["sets"].startswith("a finding against the finger fix") and "no stop limit is raised" in res["sets"]
    # beside the crossing, in the same sense: the basis pair over the window (the synthetic C1 - C2 alternates, mean 0)
    # and the same session's control pair (control_again minus control, +3.0); both named in the crossing's text
    cell = res["stages"]["prediction_tactile"]
    assert abs(cell["basis_pair_difference"]) < 1e-9 and abs(cell["session_pair_difference"] - 3.0) < 1e-9
    beside = ("beside it, in the same sense, C1 minus C2 over model steps 481,900 to 482,800 +0.0 and control_again "
              "minus control +3.0")
    assert beside in cell["why"] and beside in res["verdict_reason"]
    assert "beside it" not in res["stages"]["prediction_vision"]["why"]          # a stage within its limit: no text
    # the crossing's label: +30.0 lies above both control pairs (+0.0 and +3.0); the verdict stays fail
    assert cell["crossing_label"] == "beyond both control pairs" and abs(cell["crossing_read_against"] - 3.0) < 1e-9
    assert "the crossing reads 'beyond both control pairs'" in res["verdict_reason"]
    assert "crossing_label" not in res["stages"]["prediction_vision"]
    # without control_again (a development run: a measurement run without it is refused) the session pair is named as
    # not measured; the one-record limit is 26.3, the difference +30.0
    code, res, job, made, ck = _job(lo, tmp_path / "a2", plans, steps=100, arms="control,finger")
    assert "control_again minus control not measured" in res["stages"]["prediction_tactile"]["why"]
    assert res["stages"]["prediction_tactile"]["crossing_label"] == (
        "beyond the basis pair; the session pair (control_again minus control) was not measured")
    # a reading stop of the finger arm (its gross line) is a finding; the job runs control_again
    plans["finger"] = dict(lines=_lines("finger", rise=dict(proprioception=2000.0)), args_extra=pkg)
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans)
    assert [p.name for p in made] == ["control", "finger", "control_again"] and made[1].killed
    assert res["verdict"] == "fail" and res["stages"]["emergency_lines_finger"]["met"] is False
    assert "gross line" in res["stages"]["emergency_lines_finger"]["why"]
    # a reading stop of the control arm ends the job: invalid
    plans["finger"] = dict(lines=_lines("finger"), args_extra=pkg)
    plans["control"] = dict(lines=_lines("control", extra={482100: dict(nan_steps=3)}))
    code, res, job, made, ck = _job(lo, tmp_path / "c", plans)
    assert [p.name for p in made] == ["control"] and res["verdict"] == "invalid"
    assert "non_finite_update" in res["verdict_reason"]
    # a machine stop ends the job: invalid
    plans["control"] = dict(lines=_lines("control"))
    code, res, job, made, ck = _job(lo, tmp_path / "d", plans, machine=lambda: ((80.0, 84.4, 1.0), []))
    assert [p.name for p in made] == ["control"] and res["verdict"] == "invalid" and "commit charge" in res["verdict_reason"]
    # an invalid row's sets text names its own validity problems, not a session pair it never measured
    assert res["sets"].startswith("nothing is set from this row") and "commit charge" in res["sets"]
    assert "session" not in res["sets"]
    # an arm that ends in an error ends the job
    plans["finger"] = dict(lines=_lines("finger")[:2], code=1, args_extra=pkg, problems=["the trainer returned 1"])
    code, res, job, made, ck = _job(lo, tmp_path / "e", plans)
    assert [p.name for p in made] == ["control", "finger"] and res["verdict"] == "invalid"


def test_the_machine_stops_end_the_arm_and_the_job(lo, tmp_path):
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    plans = dict(control=dict(lines=_lines("control", steps=100)),
                 finger=dict(lines=_lines("finger", steps=100), args_extra=pkg),
                 control_again=dict(lines=_lines("control_again", steps=100)))

    # measurement runs of 100 model steps: the synthetic limits file's window set to one record (the job refuses a
    # declared length other than the limits file's window)
    def one(limits):
        limits["window"]["n"] = 1
    # an arm that never ends: its deadline (2 x (152.1 + 352.9) = 1,010 s for 100 model steps) ends it and the job
    hung = dict(plans, control=dict(lines=[], hang=True))
    code, res, job, made, ck = _job(lo, tmp_path / "a", hung, steps=100, declared=100, limits_edit=one)
    assert [p.name for p in made] == ["control"] and made[0].killed and res["verdict"] == "invalid"
    assert "deadline of 1010 s" in res["verdict_reason"]
    # a python process on the graphics card that is not the arm's
    foreign = lambda: dict(apps=[dict(pid=4242, name="C:/x/python.exe")], memory_used_mib=[3000])  # noqa: E731
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans, steps=100, declared=100, gpu=foreign, limits_edit=one)
    assert [p.name for p in made] == ["control"] and made[0].killed and "not the arm's" in res["verdict_reason"]
    # the card's memory at its line
    full = lambda: dict(apps=[], memory_used_mib=[8100])  # noqa: E731
    code, res, job, made, ck = _job(lo, tmp_path / "c", plans, steps=100, declared=100, gpu=full, limits_edit=one)
    assert res["verdict"] == "invalid" and "at or above 8100" in res["verdict_reason"]
    # another trainer started beside the arm
    code, res, job, made, ck = _job(lo, tmp_path / "d", plans, steps=100, declared=100, limits_edit=one,
                                    during=lambda own, root=None: ["process 99 runs the trainer"])
    assert [p.name for p in made] == ["control"] and made[0].killed and "runs the trainer" in res["verdict_reason"]


def test_the_comparison_reads_the_session_pair_the_settings_and_the_package(lo, tmp_path):
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    # the session's control pair beyond the declared normal variation (vision's check at 10 records: t at 1 - 0.01 / 8 on
    # 10 df, 4.01, x 40 / sqrt(10) = 50.7 in 1,000): invalid, and nothing is set
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger"), args_extra=pkg),
                 control_again=dict(lines=_lines("control_again", rise=dict(vision=80.0))))
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans)
    assert res["verdict"] == "invalid" and "session's control pair" in res["verdict_reason"]
    assert res["sets"].startswith("nothing is set from this row") and "session's control pair" in res["sets"]
    # a setting beyond the package's three: invalid
    plans["control_again"] = dict(lines=_lines("control_again"))
    plans["finger"] = dict(lines=_lines("finger"), args_extra=dict(pkg, babble_noise=0.1))
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans)
    assert res["verdict"] == "invalid" and "babble_noise" in res["verdict_reason"]
    # the package's replacement missing from the finger arm's stream: invalid
    no_replacement = [x for x in _lines("finger") if x.get("type") != "hand_calibration_replacement"]
    plans["finger"] = dict(lines=no_replacement, args_extra=pkg)
    code, res, job, made, ck = _job(lo, tmp_path / "c", plans)
    assert res["verdict"] == "invalid" and "hand calibration replacements" in res["verdict_reason"]
    # a development run (one record) passes on its plumbing alone
    plans["finger"] = dict(lines=_lines("finger", steps=100, rise=dict(tactile=30.0)), args_extra=pkg)
    plans["control"] = dict(lines=_lines("control", steps=100))
    plans["control_again"] = dict(lines=_lines("control_again", steps=100))
    code, res, job, made, ck = _job(lo, tmp_path / "d", plans, steps=100)
    assert res["development"] is True and res["verdict"] == "pass" and "prediction_tactile False" in res["verdict_reason"]
    # a development run's session pair beyond its one-record check (vision: 4.01 x 40 = 160 in 1,000; the rise of 200
    # stays under the gross line of 670 + 5 x 40) is a finding, never the development row's verdict; it sets nothing
    plans["control_again"] = dict(lines=_lines("control_again", steps=100, rise=dict(vision=200.0)))
    code, res, job, made, ck = _job(lo, tmp_path / "e", plans, steps=100)
    assert res["verdict"] == "pass" and len(res["findings"]) == 1 and "session's control pair" in res["findings"][0]
    assert "session's control pair" in res["verdict_reason"] and res["sets"].startswith("nothing: a development run")


def test_a_crossing_in_the_record_written_as_the_arm_ends(lo, tmp_path):
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    # the finger arm's last record crosses its gross line in the poll that ends the arm: a finding, control_again runs
    finger = _lines("finger")
    finger[-1]["prediction_audit_per_body"]["senses"]["tactile"]["without_body_2"]["ratio"] = 2.0
    plans = dict(control=dict(lines=_lines("control")),
                 finger=dict(lines=finger, args_extra=pkg, last_with_exit=True),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans)
    row = [c for c in job["children"] if c["arm"] == "finger"][0]
    assert row["stop_kind"] == "reading" and row["processes_ended"] == 0 and not made[1].killed
    assert [p.name for p in made] == ["control", "finger", "control_again"]
    assert res["verdict"] == "fail" and res["stages"]["emergency_lines_finger"]["met"] is False
    assert "gross line" in res["stages"]["emergency_lines_finger"]["why"]
    # the same crossing in the control arm's last record ends the job: invalid
    control = _lines("control")
    control[-1]["prediction_audit_per_body"]["senses"]["tactile"]["without_body_2"]["ratio"] = 2.0
    plans = dict(control=dict(lines=control, last_with_exit=True), finger=dict(lines=_lines("finger"), args_extra=pkg),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans)
    assert [p.name for p in made] == ["control"] and res["verdict"] == "invalid" and "gross line" in res["verdict_reason"]


def test_the_original_checkpoint_is_read_before_and_after(lo, tmp_path):
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger"), args_extra=pkg),
                 control_again=dict(lines=_lines("control_again")))
    original = _Arm.poll

    def touching(self):                                          # an arm that writes the original's optimizer file
        code = original(self)
        if code is not None and self.name == "finger":
            ck = Path(self.cmd[self.cmd.index("--checkpoint-copy") + 1]).parents[3] / "Ckpt" / "ckpt_step_481800"
            (ck / "optimizer.pt").write_bytes(b"changed")
        return code
    _Arm.poll = touching
    try:
        code, res, job, made, ck = _job(lo, tmp_path, plans)
    finally:
        _Arm.poll = original
    assert job["original_changed"] == ["optimizer.pt"] and res["verdict"] == "invalid"


def test_the_job_refuses_another_limits_file_code_or_length(lo, tmp_path, monkeypatch):
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger"), args_extra=pkg),
                 control_again=dict(lines=_lines("control_again")))
    with pytest.raises(SystemExit):
        _job(lo, tmp_path / "a", plans, steps=750)
    # a code tree whose trainer code hash is not the admission record's: refused, no arm started
    (tmp_path / "other_code").mkdir()
    with pytest.raises(SystemExit) as refused:
        _job(lo, tmp_path / "c", plans, code_tree=tmp_path / "other_code")
    assert "trainer code" in str(refused.value) and not (tmp_path / "c" / "job" / "arm_control").exists()
    # a command file whose sha256 is not the declared one: refused
    monkeypatch.setitem(lo.COMMANDS, "finger", (lo.COMMANDS["finger"][0], "0" * 64))
    with pytest.raises(SystemExit) as refused:
        _job(lo, tmp_path / "f", plans)
    assert "command file of arm finger" in str(refused.value)
    monkeypatch.undo()
    # a limits file other than the declared one (an existing file, another sha256): a problem; the declared one: none
    held = tmp_path / "held_limits.json"
    held.write_text(json.dumps(dict(senses=dict(tactile=dict(by_records={"8": {}})))), encoding="utf-8")
    assert "not the declared" in lo.limits_problems(held, "0" * 64, 800)[1][0]
    assert lo.limits_problems(held, lo.sha256_file(held), 800)[1] == []
    # a checkpoint whose model.safetensors is not 481,800's: refused before any copy, no arm started
    with pytest.raises(SystemExit) as refused:
        _job(lo, tmp_path / "m", plans, expected_model="0" * 64)
    assert "not 481,800's" in str(refused.value)
    assert not (tmp_path / "m" / "job" / "arm_control").exists()
    with pytest.raises(SystemExit):
        lo.parse_arms("finger,control")
    assert lo.parse_arms("control,finger") == ["control", "finger"]
    limits, problems = lo.limits_problems(tmp_path / "missing.json", "0" * 64, 800)
    assert problems and "sha256" in problems[0]


def test_the_discard_never_leaves_the_jobs_folder(lo, tmp_path):
    job = tmp_path / "job"
    a = job / "arm_control"
    (a / "checkpoint_copy").mkdir(parents=True)
    (a / "checkpoint_copy" / "model.safetensors").write_bytes(b"x" * 10)
    (a / "run").mkdir(parents=True)
    (a / "run" / "replay_training_stats.jsonl").write_text("{}\n", encoding="utf-8")
    d = lo.discard_weights(job, ["control"])
    assert [Path(r["path"]).name for r in d["removed"]] == ["checkpoint_copy"] and d["bytes"] == 10
    assert (a / "run" / "replay_training_stats.jsonl").is_file()
    with pytest.raises(SystemExit):
        lo.cmd_discard(argparse.Namespace(scratch=str(tmp_path / "nojob")))


def test_the_limit_is_read_at_its_edge(lo, tmp_path):
    # the declared limit, never widened: a rise just above it is a finding, just below it is not
    c1, c2, ll = control_runs()
    line = lo.compute_limits(dict(C1=c1, C2=c2), dict(C1=c1, C2=c2, L=ll))["senses"]["tactile"]["by_records"]["10"]["limit"]
    pkg = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")
    for factor, verdict in ((1.05, "fail"), (0.95, "pass")):
        plans = dict(control=dict(lines=_lines("control")),
                     finger=dict(lines=_lines("finger", rise=dict(tactile=factor * line)), args_extra=pkg),
                     control_again=dict(lines=_lines("control_again")))
        code, res, job, made, ck = _job(lo, tmp_path / str(factor), plans)
        assert res["verdict"] == verdict, (factor, res["verdict_reason"])
        assert abs(res["stages"]["prediction_tactile"]["limit"] - line) < 1e-9


# ------------------------------------------------------------------------------------------------ the review of Oct 9
PKG = dict(finger_joint_servo="on", hand_response_calibration="refit", replace_saved_hand_calibration="record")


def test_the_crossing_label_reads_both_control_pairs(lo):
    # FIX 2: a difference no larger than the larger of the two signed pair numbers is not distinguished from normal
    # variation; one above both is beyond both control pairs; the label never decides the verdict
    assert lo.crossing_label(36.0, 37.7, 5.0) == (lo.NOT_DISTINGUISHED, 37.7)
    assert lo.crossing_label(36.0, 2.0, 40.0) == (lo.NOT_DISTINGUISHED, 40.0)
    assert lo.crossing_label(36.0, 27.7, 5.0) == (lo.BEYOND_BOTH, 27.7)
    # a control pair below 0 cannot exceed a crossing's difference (both read signed)
    assert lo.crossing_label(30.0, -40.0, -35.0) == (lo.BEYOND_BOTH, -35.0)
    # the session pair not measured: the basis pair alone, and the label says so
    assert lo.crossing_label(30.0, 35.0, None) == (
        lo.NOT_DISTINGUISHED + "; the session pair (control_again minus control) was not measured", 35.0)
    assert lo.crossing_label(40.0, 35.0, None) == (
        "beyond the basis pair; the session pair (control_again minus control) was not measured", 35.0)
    assert lo.crossing_label(40.0, None, None) == ("not compared: neither control pair was measured", None)


def test_a_crossing_is_labelled_and_still_fails(lo, tmp_path):
    # the finger arm's tactile difference +10.0 lies above its limit (8.3) and no higher than the session pair's +12.0
    # (inside its check, 12.7): labelled not distinguished from normal variation; the verdict stays fail
    plans = dict(control=dict(lines=_lines("control")),
                 finger=dict(lines=_lines("finger", rise=dict(tactile=10.0)), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again", rise=dict(tactile=12.0))))
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans)
    cell = res["stages"]["prediction_tactile"]
    assert res["verdict"] == "fail" and code == 1 and cell["met"] is False
    assert cell["crossing_label"] == lo.NOT_DISTINGUISHED and abs(cell["crossing_read_against"] - 12.0) < 1e-9
    assert ("the crossing reads 'not distinguished from normal variation' (+10.0 is no larger than +12.0"
            in res["verdict_reason"])
    assert res["sets"].startswith("a finding against the finger fix")
    # a development run without control_again: read against the basis pair alone (set here to +35.0), the session pair
    # named as not measured; the one-record limit is 26.3, the difference +30.0

    def basis35(limits):
        limits["senses"]["tactile"]["mean_D"] = 35.0
    plans = dict(control=dict(lines=_lines("control", steps=100)),
                 finger=dict(lines=_lines("finger", steps=100, rise=dict(tactile=30.0)), args_extra=PKG))
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans, steps=100, arms="control,finger", limits_edit=basis35)
    cell = res["stages"]["prediction_tactile"]
    assert cell["met"] is False and cell["crossing_label"] == (
        lo.NOT_DISTINGUISHED + "; the session pair (control_again minus control) was not measured")
    assert abs(cell["basis_pair_difference"] - 35.0) < 1e-9 and cell["session_pair_difference"] is None


def test_the_halves_and_the_named_loss_beside_each_prediction_stage(lo, tmp_path):
    # FIX 6: a rise carried by records 1 to 5 (+44.0 in tactile, 0 after) reads +22.0 over the 10 records, and the stage
    # names +44.0 over records 1 to 5 and +0.0 over records 6 to 10. FIX 5: 22.0 exceeds tactile's named loss of 20:
    # gof yes; audio's +40.0 lies above its limit (25.0) and under its named loss of 50: gof no
    finger = _lines("finger", rise=dict(audio=40.0))
    for rec in finger:
        if rec.get("type") == "mimo" and rec["step"] <= 482300:
            rec["prediction_audit_per_body"]["senses"]["tactile"]["without_body_2"]["ratio"] += 0.044
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=finger, args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path, plans)
    t = res["stages"]["prediction_tactile"]
    assert abs(t["difference"] - 22.0) < 1e-9 and t["met"] is False
    assert t["first_half"]["records"] == "1 to 5" and abs(t["first_half"]["mean"] - 44.0) < 1e-9
    assert t["second_half"]["records"] == "6 to 10" and abs(t["second_half"]["mean"]) < 1e-9
    assert t["first_half"]["model_steps"] == [481900, 482000, 482100, 482200, 482300]
    assert "over records 1 to 5 +44.0, over records 6 to 10 +0.0" in t["why"]
    v = res["stages"]["prediction_vision"]
    assert v["met"] is True and abs(v["first_half"]["mean"]) < 1e-9 and abs(v["second_half"]["mean"]) < 1e-9
    readings = {r["name"]: r for r in res["readings"]}
    rt = readings["tactile prediction ratio, finger arm minus control arm"]
    ra = readings["audio prediction ratio, finger arm minus control arm"]
    assert rt["gof_material"] == "yes" and ra["gof_material"] == "no"
    assert "named loss in tactile is 20 in 1,000" in rt["baseline"] and "margin 57 in 1,000" in rt["baseline"]
    assert "named loss in audio is 50 in 1,000" in ra["baseline"] and "margin 62 in 1,000" in ra["baseline"]
    assert readings["vision prediction ratio, finger arm minus control arm"]["gof_material"] == "no"
    assert "records 1 to 5 +44.0" in rt["meaning"] and "the crossing reads" in rt["meaning"]


def test_the_replay_lesson_is_reported_per_arm_and_record(lo, tmp_path):
    # NOTE 1: the finger arm's replay lesson draws nothing in its first record (replay_steps 0, replay_loss null) and
    # starts in its second; the result reports both fields per arm and record beside the stages
    finger = _lines("finger")
    recs = [r for r in finger if r.get("type") == "mimo"]
    recs[0].update(replay_loss=None, next_frame_norm=dict(replay_steps=0))
    recs[1].update(replay_loss=0.2, next_frame_norm=dict(replay_steps=35))
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=finger, args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path, plans)
    assert res["verdict"] == "pass", res["verdict_reason"]
    f, c = res["replay_lesson"]["arms"]["finger"], res["replay_lesson"]["arms"]["control"]
    assert f["model_steps"][:2] == [481900, 482000] and f["replay_steps"][:3] == [0, 35, 50]
    assert f["replay_loss"][0] is None and abs(f["replay_loss"][1] - 0.2) < 1e-12
    assert c["replay_steps"] == [50] * 10 and abs(c["replay_loss"][0] - 0.09) < 1e-12
    assert "129 entries recorded after the switch" in res["replay_lesson"]["meaning"]


def test_after_a_finger_stop_the_job_waits_for_the_preconditions(lo, tmp_path):
    # FIX 3: the finger arm stopped on its gross line; its tree is ended and waited for (60 s at most); before
    # control_again the preconditions read busy at the first two readings, then clear, and the job goes on
    reads = iter([[], [], [], ["process 4243 runs a learning-on arm"], ["the graphics card holds 900 MiB in use"]])
    plans = dict(control=dict(lines=_lines("control")),
                 finger=dict(lines=_lines("finger", rise=dict(proprioception=2000.0)), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans, preconditions_fn=lambda: next(reads, []))
    assert [p.name for p in made] == ["control", "finger", "control_again"] and made[1].killed
    assert made[1].waited == lo.END_WAIT_S == 60.0
    row = [c for c in job["children"] if c["arm"] == "finger"][0]
    assert row["end_wait"]["alive_after_wait"] == [] and row["end_wait"]["wait_limit_s"] == 60.0
    w = job["precondition_waits"]
    assert len(w) == 1 and w[0]["before_arm"] == "control_again" and w[0]["polls"] == 3 and w[0]["cleared"] is True
    assert w[0]["first"] == ["process 4243 runs a learning-on arm"] and w[0]["limit_s"] == 120.0
    assert res["verdict"] == "fail" and res["stages"]["emergency_lines_finger"]["met"] is False
    # preconditions still busy when the 120 s have passed end the job: invalid, the wait recorded
    busy = iter([[], [], []])
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans,
                                    preconditions_fn=lambda: next(busy, ["process 4243 runs a learning-on arm"]))
    assert [p.name for p in made] == ["control", "finger"] and res["verdict"] == "invalid"
    assert "before arm control_again: process 4243 runs a learning-on arm" in res["verdict_reason"]
    w = job["precondition_waits"]
    assert len(w) == 1 and w[0]["cleared"] is False and w[0]["seconds"] >= 120


class _FakePs:
    """psutil's Process(pid) (nice, create_time) and NoSuchProcess on dicts of priority classes and creation times (no
    real process). Process(pid) is a new lookup: the process that holds the id now (created[pid]). listed(pid, created):
    the object a poll's listing holds, its creation time kept from the listing as psutil keeps it; its setter refuses
    an id that a process created at another time holds now (psutil's own check of a reused id, NoSuchProcess), and a
    gone id raises NoSuchProcess."""
    NORMAL_PRIORITY_CLASS = 32

    class NoSuchProcess(Exception):
        pass

    def __init__(self, classes, gone=(), created=None):
        self.classes, self.gone, self.created = dict(classes), set(gone), dict(created or {})

    def _proc(self, pid, kept):
        fake = self

        class _P:
            def __init__(self):
                self.pid = pid

            def create_time(self):
                return kept

            def nice(self, value=None):
                if pid in fake.gone:
                    raise fake.NoSuchProcess(pid)
                if value is None:
                    return fake.classes[pid]
                if fake.created.get(pid, 1.0) != kept:
                    raise fake.NoSuchProcess(pid)            # psutil: "its PID has been reused"
                fake.classes[pid] = int(value)
        return _P()

    def Process(self, pid):  # noqa: N802 - psutil's name
        if pid in self.gone:
            raise self.NoSuchProcess(pid)
        return self._proc(pid, self.created.get(pid, 1.0))

    def listed(self, pid, created=None):
        return self._proc(pid, self.created.get(pid, 1.0) if created is None else created)


def test_the_arms_tree_is_set_to_normal_priority_and_out_of_power_throttling(lo, tmp_path):
    # FIX 4 on stubs: two processes below normal (16384) raised to normal (32), one already normal left as it is, one
    # that ended between the listing and the setting counted, never an error; each live one opted out of throttling
    fake = _FakePs({10: 16384, 11: 16384, 12: 32}, gone={13})
    opted = []
    got = lo.set_normal_priority([fake.listed(pid) for pid in (10, 11, 12, 13)], ps=fake,
                                 opt_out=lambda pid: opted.append(pid) or dict(opted_out=True))
    assert fake.classes == {10: 32, 11: 32, 12: 32} and opted == [10, 11, 12]
    assert got[10] == dict(priority_before=16384, priority=32, throttling=dict(opted_out=True))
    assert got[12]["priority_before"] == 32 and got[13] == dict(ended=True)
    # the arm's row: two raised, one process gone, and an opt-out that finds its process gone (Windows error 87)
    row = dict(priority=dict(polls=0, raised_to_normal=0, ended_between=0, first_read={}, errors=[]),
               power_throttling=dict(opted_out=[], errors=[]))
    lo.note_priority(row, got)
    lo.note_priority(row, {10: dict(priority_before=32, priority=32,
                                    throttling=dict(opted_out=False, error=87, ended=True))})
    assert row["priority"]["polls"] == 2 and row["priority"]["raised_to_normal"] == 2
    assert row["priority"]["ended_between"] == 2 and row["power_throttling"]["opted_out"] == ["10", "11", "12"]
    assert row["priority"]["first_read"] == {"10": 16384, "11": 16384, "12": 32}
    # the job sets the arm's tree at every poll while the arm runs and records it in the arm's row; the listing names
    # the root and two children (the second re-check's note 11: a change that set only the root's class leaves the
    # children below normal), each set through set_normal_priority on the fake psutil
    fake, opted, calls = _FakePs({}), [], []

    def listing(proc):
        tree = [proc.pid, 10 * proc.pid + 1, 10 * proc.pid + 2]
        for pid in tree:
            fake.classes.setdefault(pid, 16384)
        calls.append(tree)
        return [fake.listed(pid) for pid in tree]

    def priority(procs):
        return lo.set_normal_priority(procs, ps=fake, opt_out=lambda pid: opted.append(pid) or dict(opted_out=True))
    plans = dict(control=dict(lines=_lines("control", steps=100)),
                 finger=dict(lines=_lines("finger", steps=100), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again", steps=100)))
    code, res, job, made, ck = _job(lo, tmp_path, plans, steps=100, priority=priority, listing=listing)
    for arm, child in zip(made, job["children"]):
        tree = [arm.pid, 10 * arm.pid + 1, 10 * arm.pid + 2]
        assert child["priority"]["polls"] == len([c for c in calls if c[0] == arm.pid]) >= 2
        assert all(fake.classes[pid] == 32 for pid in tree), (arm.name, fake.classes)
        assert child["priority"]["first_read"] == {str(pid): 16384 for pid in tree}
        assert child["priority"]["raised_to_normal"] == 3
        assert sorted(child["power_throttling"]["opted_out"]) == sorted(str(pid) for pid in tree)
        assert child["priority"]["below_normal_after_first_poll"] == dict(count=0, processes=[])


def test_the_commit_charge_is_recorded_just_before_each_arm(lo, tmp_path):
    # NOTE 13: the commit charge, the commit limit and the free memory read just before each arm, in the job record
    plans = dict(control=dict(lines=_lines("control", steps=100)),
                 finger=dict(lines=_lines("finger", steps=100), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again", steps=100)))
    code, res, job, made, ck = _job(lo, tmp_path, plans, steps=100)
    assert sorted(job["machine_before_arm"]) == ["control", "control_again", "finger"]
    for child in job["children"]:
        m = child["machine_before"]
        assert (m["committed_gb"], m["limit_gb"], m["free_ram_gb"]) == (20.0, 84.4, 30.0)
        assert job["machine_before_arm"][child["arm"]] == m
    nothing = lo.machine_reading(lambda: (None, [("C:/", 9.0)]))
    assert nothing["error"] == "commit_guard read nothing" and nothing["short_drives"] == [["C:/", 9.0]]


def test_a_lifted_pause_stops_the_arm(lo, tmp_path):
    # NOTE 14: supervisor_control.json re-read at every poll; a lifted pause is a machine stop naming its cause
    import codecs
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger"), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans, paused=lambda: (False, "cmd 'run'"))
    assert [p.name for p in made] == ["control"] and made[0].killed and res["verdict"] == "invalid"
    assert "the supervisor's pause was lifted (cmd 'run')" in res["verdict_reason"]
    assert job["children"][0]["stop_kind"] == "machine"
    # the file read as training_continuation.pause_reason reads it
    f = tmp_path / "supervisor_control.json"
    assert lo.supervisor_paused(f) == (False, "supervisor_control.json is missing")
    f.write_text('{"cmd": "pause", "repair_run": "x"}', encoding="utf-8")
    assert lo.supervisor_paused(f) == (True, "cmd 'pause'")
    f.write_text('{"cmd": "run"}', encoding="utf-8")
    assert lo.supervisor_paused(f) == (False, "cmd 'run'")
    f.write_text('{"archiver": "off"}', encoding="utf-8")
    assert lo.supervisor_paused(f)[0] is True
    f.write_text('{"cmd": "pau', encoding="utf-8")
    assert lo.supervisor_paused(f)[0] is True
    # a byte-order mark: pause_reason reads UTF-8, the JSON does not parse, and it reads paused (re-check note 2)
    f.write_bytes(codecs.BOM_UTF8 + b'{"cmd": "pause"}')
    held, seen = lo.supervisor_paused(f)
    assert held is True and "unreadable" in seen


def test_the_other_trainer_rule_reads_any_trial_trainer_and_the_arms_own_children(lo):
    # NOTE 8: any *_trial_trainer.py that is not a test file is a trainer; a process whose parents include the arm's
    # root process is the arm's own, also when it started after the arm's tree was listed
    procs = [dict(pid=1, cmdline=["python.exe", "-B", "C:/x/persistent_memory_trial_trainer.py", "arm"]),
             dict(pid=3, cmdline=["python.exe", "C:/x/test_alpha_trial_trainer.py"]),
             dict(pid=4, cmdline=["python.exe", "-B", "-u", "C:/w/learning_on_arm_trial_trainer.py", "arm"]),
             dict(pid=5, cmdline=["python.exe", "-m", "mimo_bridge.train_mimo", "--out", "x"]),
             dict(pid=6, cmdline=["python.exe", "C:/w/update_replay.py", "arm"]),
             dict(pid=7, cmdline=["python.exe", "-c", "from multiprocessing.spawn import spawn_main"]),
             dict(pid=8, cmdline=["python.exe", "C:/x/wait_no_trainer.py"]),
             dict(pid=900, cmdline=["python.exe", "-B", "-u", "C:/w/learning_on_arm_trial_trainer.py", "arm"])]
    got = lo.other_trainers({900}, 900, processes=procs, parents=lambda pid: [900] if pid == 4 else [])
    assert got == ["process 1 runs a trial trainer (persistent_memory_trial_trainer.py)",
                   "process 5 runs the trainer", "process 6 runs another arm (update_replay.py)"]
    got = lo.other_trainers({900}, 900, processes=procs, parents=lambda pid: [])
    assert "process 4 runs a trial trainer (learning_on_arm_trial_trainer.py)" in got
    # before an arm: the same suffix rule beside the replay's preconditions
    pre = lo.preconditions(processes=procs, control=dict(cmd="pause"), continuation=dict(trainer_pids=[]),
                           gpu=dict(apps=[], memory_used_mib=[0]))
    assert "process 1 runs a trial trainer (persistent_memory_trial_trainer.py)" in pre
    assert {"process 4 runs a learning-on arm", "process 5 runs the trainer", "process 6 runs a replay arm"} <= set(pre)
    assert not any(x.startswith(("process 3 ", "process 7 ", "process 8 ")) for x in pre)


def test_commit_guard_is_put_on_the_path_once(lo, monkeypatch):
    # NOTE 18: the job's 10-second machine reading puts the code tree on sys.path once, not once per poll
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(lo.SOURCE)])
    for _ in range(3):
        lo._commit_guard()
    assert sys.path.count(str(lo.SOURCE)) == 1


def test_the_job_refuses_a_relative_scratch(lo, tmp_path, monkeypatch):
    # NOTE 6: a relative --scratch is refused before anything is read or written (the arms' working folder is the code
    # tree, so a relative path would name another folder there)
    monkeypatch.chdir(tmp_path)
    a = argparse.Namespace(scratch="rel_job", out=str(tmp_path / "r.json"), steps=1000, declared_steps=1000,
                           arms="control,finger,control_again", limits=str(tmp_path / "none.json"),
                           limits_sha256="0" * 64, deadline_s=0.0)
    with pytest.raises(SystemExit) as refused:
        lo.cmd_job(a, popen=lambda *x, **k: None, contain=False, checkpoint=tmp_path / "nock",
                   preconditions_fn=lambda: [], supervisor_env=dict(PATH="x"), code_tree=tmp_path,
                   expected_model="0" * 64)
    assert "relative path" in str(refused.value) and not (tmp_path / "rel_job").exists()


def test_the_job_refuses_a_measurement_without_control_again_or_another_declared_length(lo, tmp_path):
    # NOTE 10: a measurement run (steps equal to the declared steps) needs control_again; the declared length must be
    # the limits file's window (10 records, 1,000 model steps); both refused before any copy
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger"), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))
    with pytest.raises(SystemExit) as refused:
        _job(lo, tmp_path / "a", plans, arms="control,finger")
    assert "needs arm control_again" in str(refused.value)
    assert not (tmp_path / "a" / "job" / "arm_control").exists()
    with pytest.raises(SystemExit) as refused:
        _job(lo, tmp_path / "b", plans, steps=800, declared=800)
    assert "800 model steps is not the limits file's window of 10 records (1000 model steps)" in str(refused.value)
    assert not (tmp_path / "b" / "job" / "arm_control").exists()
    # the comparison of an existing folder reads the same two problems as validity problems: invalid, its causes named
    assert lo.configuration_problems(dict(window=dict(n=10)), 1000, 1000, ["control", "finger", "control_again"]) == []
    assert lo.configuration_problems(dict(window=dict(n=10)), 100, 1000, ["control", "finger"]) == []


def test_the_discard_sizes_without_following_a_reparse_point_and_names_its_errors(lo, tmp_path, monkeypatch):
    # NOTE 7: a target's size read with os.walk(followlinks=False), never into or through a reparse point
    cc = tmp_path / "sizing" / "checkpoint_copy"
    (cc / "inner").mkdir(parents=True)
    (cc / "m.bin").write_bytes(b"m" * 10)
    (cc / "inner" / "w.bin").write_bytes(b"w" * 5)
    assert lo.tree_bytes(cc) == 15
    # a folder read as a reparse point is neither entered nor counted (the predicate given here: the tests' drive D:
    # holds no junction)
    real = lo.is_reparse_point
    monkeypatch.setattr(lo, "is_reparse_point", lambda p: str(p).endswith("inner") or real(p))
    assert lo.tree_bytes(cc) == 10
    monkeypatch.setattr(lo, "is_reparse_point", real)
    # real junctions: test_the_discard_sizing_on_real_junctions_on_c (D:, where pytest's folders lie, takes none)
    # every path the discard cannot remove is named in the job's result
    import shutil
    original = shutil.rmtree

    def held(path, *args, **kwargs):
        if str(path).endswith("checkpoint_copy"):
            raise PermissionError("held by an ended process")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(shutil, "rmtree", held)
    plans = dict(control=dict(lines=_lines("control")), finger=dict(lines=_lines("finger"), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))
    code, res, job, made, ck = _job(lo, tmp_path / "j", plans)
    monkeypatch.setattr(shutil, "rmtree", original)
    assert res["verdict"] == "pass", res["verdict_reason"]
    assert len(res["discard_problems"]) == 3
    assert all("checkpoint_copy: PermissionError: held by an ended process" in p for p in res["discard_problems"])
    assert res["discard"] == job["discarded"] and res["discard"]["bytes"] > 0


def test_a_stopped_trees_processes_are_waited_for(lo):
    # FIX 3 on fakes: the tree listed before the end is handed to psutil.wait_procs with the 60 s limit; a process still
    # alive after it is named in the record
    class _Proc:
        def __init__(self, pid):
            self.pid = pid

    class _Root:
        pid, killed = 41, False

        def kill(self):
            self.killed = True
    calls = []

    def wait_procs(procs, timeout):
        calls.append(([p.pid for p in procs], timeout))
        return procs[:1], procs[1:]
    root = _Root()
    got = lo.end_and_wait(root, tree=[_Proc(42), _Proc(43), _Proc(41)], wait_procs=wait_procs)
    assert calls == [([42, 43, 41], 60.0)] and root.killed
    assert got["alive_after_wait"] == [41, 43] and got["wait_limit_s"] == 60.0


# ------------------------------------------------------------------------------------------------ the second re-check
def _short_plans():
    return dict(control=dict(lines=_lines("control", steps=100)),
                finger=dict(lines=_lines("finger", steps=100), args_extra=PKG),
                control_again=dict(lines=_lines("control_again", steps=100)))


def test_the_trainer_rules_read_python_processes_only(lo):
    # FIX 1: a program that is not python (sha256sum.exe, notepad.exe, cat.exe, grep.exe) naming a *_trial_trainer.py or
    # mimo_bridge/train_mimo.py is not a trainer, while an arm runs (other_trainers) or before one (preconditions); a
    # python process doing the same is one
    others = [dict(pid=5000, cmdline=["C:/Program Files/Git/usr/bin/sha256sum.exe",
                                      "C:/w/docs/x/learning_on_arm_trial_trainer.py"]),
              dict(pid=5004, cmdline=["C:/Windows/notepad.exe", "C:/w/persistent_memory_trial_trainer.py"]),
              dict(pid=5008, cmdline=["C:/Program Files/Git/usr/bin/cat.exe",
                                      "C:/w/Source/Python/mimo_bridge/train_mimo.py"]),
              dict(pid=5012, cmdline=["C:/Program Files/Git/usr/bin/grep.exe", "-n", "x",
                                      "C:/w/Source/Python/mimo_bridge/train_mimo.py"]),
              dict(pid=5016, cmdline=["C:\\Program Files\\Git\\usr\\bin\\grep.exe",
                                      "C:/w/update_replay_trial_trainer.py"])]
    pythons = [dict(pid=6000, cmdline=["C:/Py313/python.exe", "-B", "C:/w/persistent_memory_trial_trainer.py", "arm"]),
               dict(pid=6004, cmdline=["python", "C:/w/Source/Python/mimo_bridge/train_mimo.py", "--out", "x"]),
               dict(pid=6008, cmdline=["C:\\v\\.venv_mimo\\Scripts\\python.exe", "-m", "mimo_bridge.train_mimo"])]
    assert lo.other_trainers({1}, 1, processes=others, parents=lambda p: []) == []
    held = dict(control=dict(cmd="pause"), continuation=dict(trainer_pids=[]), gpu=dict(apps=[], memory_used_mib=[0]))
    assert lo.preconditions(processes=others, **held) == []
    assert lo.other_trainers({1}, 1, processes=others + pythons, parents=lambda p: []) == [
        "process 6000 runs a trial trainer (persistent_memory_trial_trainer.py)", "process 6004 runs the trainer",
        "process 6008 runs the trainer"]
    assert lo.preconditions(processes=others + pythons, **held) == [
        "process 6004 runs the trainer", "process 6008 runs the trainer",
        "process 6000 runs a trial trainer (persistent_memory_trial_trainer.py)"]
    # the executable's file name: the command line's first entry, or psutil's name when the command line is empty
    assert lo.is_python(dict(cmdline=["C:\\Py313\\pythonw.exe", "x.py"])) and lo.is_python(dict(name="python.exe"))
    assert not lo.is_python(dict(name="cat.exe", cmdline=[])) and not lo.is_python(dict(cmdline=[]))
    assert not lo.is_python(dict(name="python.exe", cmdline=["C:/bin/cat.exe", "x_trial_trainer.py"]))
    # the job's own listing carries psutil's name beside the command line: this test's process reads as python
    me = [p for p in lo.running_processes() if p["pid"] == os.getpid()]
    assert len(me) == 1 and me[0]["name"] and lo.is_python(me[0]) and lo.is_python(dict(name=me[0]["name"]))


def test_the_settings_go_through_the_listings_objects_and_a_reused_id_is_skipped(lo):
    # NOTE 1: ids 21 and 22 now name processes created after the listing (200.0 and 300.0 against 100.0 and 150.0).
    # 21, below normal: its class is set through the listing's own object, whose setter refuses the reused id (read as
    # ended), and the process now holding the id keeps its class. 22, already normal: nothing set; the new lookup's
    # creation time differs from the listing's, so the opt-out is skipped and the mismatch recorded in the arm's row
    fake = _FakePs({21: 16384, 22: 32}, created={21: 200.0, 22: 300.0})
    opted = []
    got = lo.set_normal_priority([fake.listed(21, created=100.0), fake.listed(22, created=150.0)], ps=fake,
                                 opt_out=lambda pid: opted.append(pid) or dict(opted_out=True))
    assert got[21] == dict(ended=True) and fake.classes[21] == 16384
    assert opted == [] and got[22]["throttling"] == dict(opted_out=False, reused=True, created_at_listing=150.0,
                                                         created_now=300.0)
    row = dict(priority=dict(polls=0, raised_to_normal=0, ended_between=0, first_read={}, errors=[]),
               power_throttling=dict(opted_out=[], errors=[]))
    lo.note_priority(row, got)
    assert row["power_throttling"]["id_reused"] == ["process 22: created at 150.0 at the listing, at 300.0 at the "
                                                    "setting"]
    assert row["priority"]["first_read"] == {} and row["priority"]["ended_between"] == 1
    # the same ids listed with the creation times they hold now: set and opted out
    got = lo.set_normal_priority([fake.listed(21), fake.listed(22)], ps=fake,
                                 opt_out=lambda pid: opted.append(pid) or dict(opted_out=True))
    assert fake.classes[21] == 32 and opted == [21, 22] and got[21]["priority_before"] == 16384


def test_the_processes_still_below_normal_after_the_first_poll_are_counted(lo):
    # NOTE 1: the arm's processes read below normal (16384 below normal, 64 idle) before the setting at a poll after
    # the first, each counted once in the arm's row
    row = dict(priority=dict(polls=0, raised_to_normal=0, ended_between=0, first_read={}, errors=[]),
               power_throttling=dict(opted_out=[], errors=[]))
    ok = dict(opted_out=True)
    denied = dict(priority_before=16384, priority_error="AccessDenied: (pid=32)", throttling=ok)
    lo.note_priority(row, {31: dict(priority_before=16384, priority=32, throttling=ok), 32: denied})
    assert row["priority"]["below_normal_after_first_poll"] == dict(count=0, processes=[])
    # poll 2: 31 normal; 32 still below normal (its setting refused); 33, which the first poll did not list, idle
    lo.note_priority(row, {31: dict(priority_before=32, priority=32, throttling=ok), 32: denied,
                           33: dict(priority_before=64, priority=32, throttling=ok)})
    lo.note_priority(row, {32: denied})
    assert row["priority"]["below_normal_after_first_poll"] == dict(count=2, processes=["32", "33"])
    assert row["priority"]["polls"] == 3 and row["priority"]["errors"] == ["process 32: AccessDenied: (pid=32)"] * 3


def test_the_pause_is_read_as_the_supervisor_reads_it(lo, tmp_path, monkeypatch):
    # NOTE 2: a stat error on supervisor_control.json (its existence test) reads as held; a byte-order-marked
    # {"cmd": "run"} reads as held, as training_continuation.pause_reason reads it (UTF-8: the JSON does not parse); and
    # on every content here the job's reading equals the supervisor's (a run folder with no stop reason)
    import codecs
    from mimo_bridge.tools import training_continuation as tc
    f, run = tmp_path / "supervisor_control.json", tmp_path / "run"
    f.write_text('{"cmd": "pause"}', encoding="utf-8")
    real = os.stat

    def failing(path, *a, **k):
        if str(path).endswith("supervisor_control.json"):
            raise PermissionError(13, "simulated access error")
        return real(path, *a, **k)
    monkeypatch.setattr(os, "stat", failing)
    held, seen = lo.supervisor_paused(f)
    monkeypatch.setattr(os, "stat", real)
    assert held is True and "PermissionError" in seen
    contents = [b"", b'{"cmd": "pau', '{"cmd": "pause"}'.encode("utf-16"), b'{"archiver": "off"}', b'{"cmd": "run"}',
                b'{"cmd": "run", "archiver": "off"}', codecs.BOM_UTF8 + b'{"cmd": "pause"}',
                codecs.BOM_UTF8 + b'{"cmd": "run"}', b'{"cmd": null}', b"[]", b'"pause"', b'{"cmd": "pause"}']
    for raw in contents:
        f.write_bytes(raw)
        assert lo.supervisor_paused(f)[0] is (tc.pause_reason(tmp_path, run) is not None), raw
    f.write_bytes(codecs.BOM_UTF8 + b'{"cmd": "run"}')
    assert lo.supervisor_paused(f)[0] is True and tc.pause_reason(tmp_path, run) == "unreadable supervisor control"
    f.unlink()
    assert lo.supervisor_paused(f) == (False, "supervisor_control.json is missing")
    assert tc.pause_reason(tmp_path, run) is None


def test_the_jobs_deadline_ends_the_wait_and_no_copy_follows_it(lo, tmp_path):
    # NOTE 3: the job's deadline (5,000 s on the job's clock) passes at the reading before control_again: the wait for
    # busy preconditions stops at once, the job ends with the deadline as its cause, and control_again gets no copy
    shared = dict(t=0.0)

    def late(busy):
        calls = dict(n=0)

        def reading():
            calls["n"] += 1
            if calls["n"] <= 3:                              # at the start, before control, before finger
                return []
            shared["t"] += 10000.0                           # the reading before control_again: past the deadline
            return ["the graphics card holds 900 MiB in use"] if busy else []
        return reading
    code, res, job, made, ck = _job(lo, tmp_path / "a", _short_plans(), steps=100, preconditions_fn=late(True),
                                    clock=shared, deadline_s=5000.0)
    assert [p.name for p in made] == ["control", "finger"]
    assert job["stopped"] == "before arm control_again: the job's deadline passed"
    w = job["precondition_waits"]
    assert len(w) == 1 and w[0]["job_deadline_passed"] is True and w[0]["polls"] == 1 and w[0]["cleared"] is False
    assert not (tmp_path / "a" / "job" / "arm_control_again").exists()
    assert "the job's deadline passed" in res["verdict_reason"]
    # preconditions clear, the deadline passed: no wait, and still no copy
    shared["t"] = 0.0
    code, res, job, made, ck = _job(lo, tmp_path / "b", _short_plans(), steps=100, preconditions_fn=late(False),
                                    clock=shared, deadline_s=5000.0)
    assert [p.name for p in made] == ["control", "finger"] and job["precondition_waits"] == []
    assert job["stopped"] == "before arm control_again: the job's deadline passed"
    assert not (tmp_path / "b" / "job" / "arm_control_again").exists()


def test_a_process_left_alive_after_the_end_wait_holds_the_next_arm(lo, tmp_path):
    # NOTE 4: the finger arm stopped on its gross line and one process of its tree (4242) was still alive after the end
    # wait; before control_again it reads as a precondition still busy and is re-read until it ends
    plans = dict(control=dict(lines=_lines("control")),
                 finger=dict(lines=_lines("finger", rise=dict(proprioception=2000.0)), args_extra=PKG),
                 control_again=dict(lines=_lines("control_again")))

    def end(proc):
        proc.kill()
        return dict(ended=9, wait_limit_s=60.0, waited_s=60.0, alive_after_wait=[4242])
    checks = iter([True, True, False])
    code, res, job, made, ck = _job(lo, tmp_path / "a", plans, end=end, alive=lambda pid: next(checks, False))
    assert [p.name for p in made] == ["control", "finger", "control_again"]
    w = job["precondition_waits"]
    assert len(w) == 1 and w[0]["before_arm"] == "control_again" and w[0]["cleared"] is True and w[0]["polls"] == 3
    assert w[0]["first"] == ["process 4242 of arm finger's tree still runs after its end wait"]
    # the default reading of a process still alive (psutil.pid_exists), on this test's own process
    assert lo.still_running([os.getpid()], "finger") == [
        "process %d of arm finger's tree still runs after its end wait" % os.getpid()]
    # still alive when the wait's limit has passed: the job ends, invalid, the process named
    code, res, job, made, ck = _job(lo, tmp_path / "b", plans, end=end, alive=lambda pid: True)
    assert [p.name for p in made] == ["control", "finger"] and res["verdict"] == "invalid"
    assert ("before arm control_again: process 4242 of arm finger's tree still runs after its end wait"
            in res["verdict_reason"])


def test_a_development_runs_readings_carry_no_gof_line(lo, tmp_path):
    # NOTE 8: a development run's one-record readings carry gof "GOF threshold not yet defined", also for a tactile
    # difference (+30.0) above tactile's named loss (20), which a measurement run reads as "yes"
    plans = _short_plans()
    plans["finger"] = dict(lines=_lines("finger", steps=100, rise=dict(tactile=30.0)), args_extra=PKG)
    code, res, job, made, ck = _job(lo, tmp_path, plans, steps=100)
    assert res["development"] is True and len(res["readings"]) >= 4
    assert all(r["gof_material"] == lo.GOF_UNDEFINED == "GOF threshold not yet defined" for r in res["readings"])


def test_the_jobs_own_process_is_set_to_normal_priority(lo, tmp_path):
    # NOTE 10 on stubs: the job's own process (os.getpid() on a fake psutil) raised from below normal to normal, one
    # already normal read and left; the job records the setting beside its own throttling opt-out
    fake = _FakePs({os.getpid(): 16384})
    assert lo.own_normal_priority(ps=fake) == dict(priority_before=16384, priority=32)
    assert fake.classes[os.getpid()] == 32
    assert lo.own_normal_priority(ps=_FakePs({os.getpid(): 32})) == dict(priority_before=32, priority=32)
    called = []
    code, res, job, made, ck = _job(lo, tmp_path, _short_plans(), steps=100,
                                    own_priority=lambda: called.append(1) or dict(priority_before=16384, priority=99))
    assert called == [1] and job["own_priority"] == dict(priority_before=16384, priority=99)
    assert "power_throttling" in job


@pytest.mark.skipif(os.name != "nt", reason="Windows' priority classes and power throttling")
def test_a_real_tree_is_listed_set_to_normal_and_ended(lo):
    # NOTE 11, the fix round's real_path_check.py kept as a test: a python child started below normal that starts a
    # python grandchild, both sleeping; arm_processes lists the tree, set_normal_priority raises every process to normal
    # through the listing's objects and opts each out of power throttling, end_and_wait ends the tree and waits for it,
    # and a second setting reads every process ended. Only these dummy processes are touched.
    import subprocess
    import time
    import psutil
    child = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; subprocess.Popen([sys.executable, "
                              "'-c', 'import time; time.sleep(120)']); time.sleep(120)"],
                             creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    procs = []
    try:
        # a venv's python.exe starts the base interpreter as its child: 2 processes per python started, else 1
        base = getattr(sys, "_base_executable", sys.executable)
        launcher = os.path.normcase(sys.executable) != os.path.normcase(base)
        expected, t0 = (4 if launcher else 2), time.time()
        procs = lo.arm_processes(child)
        while len(procs) < expected and time.time() - t0 < 30.0:
            time.sleep(0.2)
            procs = lo.arm_processes(child)
        pids = [p.pid for p in procs]
        assert len(pids) == expected and pids[0] == child.pid, pids
        assert {p.pid: int(p.nice()) for p in procs} == {pid: psutil.BELOW_NORMAL_PRIORITY_CLASS for pid in pids}
        got = lo.set_normal_priority(procs)
        assert {pid: int(psutil.Process(pid).nice()) for pid in pids} == {pid: 32 for pid in pids}
        assert all(got[pid]["priority_before"] == psutil.BELOW_NORMAL_PRIORITY_CLASS for pid in pids), got
        assert all(got[pid]["throttling"] == dict(opted_out=True) for pid in pids), got
        ended = lo.end_and_wait(child, wait_s=60.0)
        assert ended["alive_after_wait"] == [] and not any(p.is_running() for p in procs), ended
        assert all(cell == dict(ended=True) for cell in lo.set_normal_priority(procs).values())
    finally:
        for p in procs:
            try:
                p.kill()                                  # psutil refuses an id another process took since
            except Exception:
                pass
        if child.poll() is None:
            child.kill()
        child.wait(timeout=60)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows'")
def test_the_discard_sizing_on_real_junctions_on_c(lo):
    # NOTE 11, the fix round's junction_check.py kept as a test, in a temporary folder under %TEMP% on C: (D:, where
    # pytest's folders lie, refuses junctions): a folder holding a 10-byte file, a junction to an outside folder holding
    # a 100-byte file, and a junction cycle back to the folder itself; both read as reparse points and tree_bytes counts
    # the 10 bytes inside only. Everything it makes is removed.
    import shutil
    import tempfile
    temp = os.environ.get("TEMP") or tempfile.gettempdir()
    base = Path(tempfile.mkdtemp(prefix="learning_on_junction_", dir=temp))
    target, outside = base / "target", base / "outside"
    links = (target / "link", target / "inner" / "cycle")
    try:
        (target / "inner").mkdir(parents=True)
        outside.mkdir()
        (target / "a.bin").write_bytes(b"a" * 10)
        (outside / "k.bin").write_bytes(b"k" * 100)
        try:
            import _winapi
            _winapi.CreateJunction(str(outside), str(links[0]))
            _winapi.CreateJunction(str(target), str(links[1]))
        except (ImportError, OSError) as exc:
            pytest.skip("no junction could be made under %s (%s: %s)" % (temp, type(exc).__name__, exc))
        assert lo.is_reparse_point(links[0]) and lo.is_reparse_point(links[1])
        assert lo.tree_bytes(target) == 10
    finally:
        for link in links:
            if lo.is_reparse_point(link):
                os.rmdir(link)                            # the junction only, never what it points to
        kept = (outside / "k.bin").is_file()
        shutil.rmtree(base, ignore_errors=True)
    assert kept and not base.exists()
