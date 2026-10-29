# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The relaunch script's --cmd-file (Oct 9, 2026; the matched comparison of the finger fix, the operator's decision
received by 11:53 PM PDT, Oct 8). The supervisor's recorded command has been the cap arm's since Oct 6 (--attention-cap
on --attention-cap-bound 3), and the relaunch's --set replaces or appends a flag but cannot remove one, so neither arm
of the comparison could be started. With --cmd-file the relaunch starts from the exact command in a file instead; the
script still chooses the checkpoint, the run folder and the bound, and the training preflight judges the command it
starts. Only the command-building functions are called here (start_command, command_file, launch_command), on
temporary folders: nothing starts or ends a process, and the script's main() is never run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tests.test_training_preflight import _experiment_setup
from mimo_bridge.tools import training_preflight as TP

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py"


def relaunch(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("relaunch_command_file_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.L, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(module.L, "stamp", lambda: "Oct 9, 2026, 1:00:00 AM PDT")
    return module


def setup(tmp_path, monkeypatch):
    """The cap experiment's two records in a temporary preflight folder (test_training_preflight._experiment_setup),
    the supervisor's state naming the cap arm's command, as training_continuation.json does since Oct 6, and the
    relaunch module with the stand-in preflight."""
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    module = relaunch(monkeypatch, tmp_path)
    as_started = lambda cmd: [module.L.PY] + list(cmd[1:])                      # noqa: E731 (the recorded form)
    state = dict(active_run=str(start.parent.parent), command=as_started(cap))
    preflight = SimpleNamespace(launch_plan=lambda cmd: TP.launch_plan(cmd, root=root, src=src))
    return SimpleNamespace(src=src, root=root, reg=reg, start=start, cap=as_started(cap), control=as_started(control),
                           fp_r=fp_r, fp_c=fp_c, module=module, state=state, preflight=preflight)


def write_command(path, cmd, bom=False):
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + json.dumps(cmd).encode("utf-8"))
    return path


def test_the_control_arms_exact_command_starts_from_its_file_where_the_recorded_cap_command_cannot(
        tmp_path, monkeypatch):
    s = setup(tmp_path, monkeypatch)
    run = s.module.run_folder("c")
    # today's path: the recorded command carries the cap, and a --set can only change its value
    recorded, sha = s.module.start_command(s.state, "", s.start)
    assert recorded == s.state["command"] and sha is None
    cmd, changes, plan = s.module.launch_command(recorded, s.start, run, [], s.preflight)
    assert TP.flags_of(cmd)["--attention-cap"] == "on"
    cmd, changes, plan = s.module.launch_command(recorded, s.start, run, [("attention-cap", "off")], s.preflight)
    assert TP.configuration(cmd, s.src)["fingerprint"] not in (s.fp_r, s.fp_c)
    assert "has no preflight record" in TP.launch_refusal(cmd, root=s.root, src=s.src, registry=s.reg)
    # the control arm's exact command, from its file: no cap flag, the control's own record, allowed as its arm
    path = write_command(tmp_path / "control_command.json", s.control)
    recorded, sha = s.module.start_command(s.state, str(path), s.start)
    assert recorded == s.control and len(sha) == 64
    cmd, changes, plan = s.module.launch_command(recorded, s.start, run, [], s.preflight)
    flags = TP.flags_of(cmd)
    assert "--attention-cap" not in flags and "--attention-cap-bound" not in flags and changes == {}
    assert flags["--resume"] == str(s.start) and flags["--out"] == str(run)
    assert flags["--checkpoint-storage"] == str(run / "checkpoints")
    assert plan["kind"] == "repair_run" and plan["fingerprint"] == s.fp_c and flags["--max-steps"] == "3000"
    assert TP.launch_refusal(cmd, root=s.root, src=s.src, registry=s.reg) is None
    # the cap arm from its own file: its record, allowed as its arm
    path = write_command(tmp_path / "cap_command.json", s.cap)
    cmd, changes, plan = s.module.launch_command(s.module.start_command(s.state, str(path), s.start)[0], s.start,
                                                 run, [], s.preflight)
    assert plan["fingerprint"] == s.fp_r and TP.launch_refusal(cmd, root=s.root, src=s.src, registry=s.reg) is None


def test_the_scripts_own_choices_and_the_set_rules_hold_for_a_command_file(tmp_path, monkeypatch):
    s = setup(tmp_path, monkeypatch)
    run = s.module.run_folder("f")
    # a file made from another run's recorded command names that run's folders and bound: the script replaces them
    other = TP.apply_sets(s.control, [("--out", str(tmp_path / "experiments/seg41c_1629")),
                                      ("--checkpoint-storage", str(tmp_path / "experiments/seg41c_1629/checkpoints")),
                                      ("--max-steps", "484675")])
    path = write_command(tmp_path / "from_another_run.json", other)
    recorded, _sha = s.module.start_command(s.state, str(path), s.start)
    flags = TP.flags_of(s.module.launch_command(recorded, s.start, run, [], s.preflight)[0])
    assert flags["--out"] == str(run) and flags["--checkpoint-storage"] == str(run / "checkpoints")
    assert flags["--max-steps"] == "3000"                                       # the record's bound, as before
    cmd, changes, _plan = s.module.launch_command(recorded, s.start, run, [("max-steps", "2500")], s.preflight)
    assert TP.flags_of(cmd)["--max-steps"] == "2500" and changes == {"--max-steps": "2500"}
    assert TP.launch_refusal(cmd, root=s.root, src=s.src, registry=s.reg) is None
    # past the bound: refused by the preflight, as for the recorded command
    cmd = s.module.launch_command(recorded, s.start, run, [("max-steps", "3001")], s.preflight)[0]
    assert "not bounded as declared" in TP.launch_refusal(cmd, root=s.root, src=s.src, registry=s.reg)
    for flag in ("resume", "--out", "checkpoint-storage"):
        with pytest.raises(SystemExit, match="refused: --set --"):
            s.module.launch_command(recorded, s.start, run, [(flag, str(tmp_path / "elsewhere"))], s.preflight)


def test_a_command_file_that_cannot_be_the_command_started_is_refused(tmp_path, monkeypatch):
    s = setup(tmp_path, monkeypatch)
    cases = {
        "not_json.json": (b"[\"python\", ", "cannot be read as JSON"),
        "an_object.json": (json.dumps(dict(command=s.control)).encode(), "as a JSON list of texts"),
        "empty.json": (b"[]", "as a JSON list of texts"),
        "a_number.json": (json.dumps(s.control + [3]).encode(), "as a JSON list of texts"),
        "another_python.json": (json.dumps(["C:/Python311/python.exe"] + s.control[1:]).encode(),
                                "not this checkout's interpreter"),
        "another_module.json": (json.dumps([s.module.L.PY, "-u", "-m", "mimo_bridge.tools.training_preflight",
                                            "check"]).encode(), "the recorded command's form"),
        "module_as_a_value.json": (json.dumps([s.module.L.PY, "--tag", "mimo_bridge.train_mimo"]).encode(),
                                   "the recorded command's form"),
        "no_module.json": (json.dumps([s.module.L.PY, "-u", "-m"]).encode(), "the recorded command's form"),
        "another_checkpoint.json": (json.dumps(TP.apply_sets(s.control, [("--resume", str(
            s.start.parent / "ckpt_step_900"))])).encode(), "but this relaunch resumes"),
        "resume_without_a_value.json": (json.dumps([a for a in s.control if a != str(s.start)]).encode(),
                                        "resumes"),
        "missing.json": (None, "cannot be read as JSON"),
    }
    for name, (data, why) in cases.items():
        path = tmp_path / name
        if data is not None:
            path.write_bytes(data)
        with pytest.raises(SystemExit, match=why):
            s.module.start_command(s.state, str(path), s.start)


def test_interpreter_options_other_than_the_recorded_form_are_refused(tmp_path, monkeypatch):
    # the independent review of Oct 9, 2026, finding F2 (its probe R2): each of these passed command_file, took the
    # control's fingerprint and was allowed by the preflight, which reads only the arguments after the module
    s = setup(tmp_path, monkeypatch)
    tail = s.control[s.control.index("mimo_bridge.train_mimo") + 1:]
    heads = {"dash_c": [s.module.L.PY, "-c", "print('not the trainer')", "-m", "mimo_bridge.train_mimo"],
             "other_module_first": [s.module.L.PY, "-u", "-m", "mimo_bridge.tools.training_preflight", "-m",
                                    "mimo_bridge.train_mimo"],
             "optimize": [s.module.L.PY, "-O", "-u", "-m", "mimo_bridge.train_mimo"],
             "without_u": [s.module.L.PY, "-m", "mimo_bridge.train_mimo"],
             "u_after_m": [s.module.L.PY, "-m", "-u", "mimo_bridge.train_mimo"]}
    for name, head in heads.items():
        path = write_command(tmp_path / (name + ".json"), head + tail)
        with pytest.raises(SystemExit, match="must start .* -u -m mimo_bridge.train_mimo, the recorded command's form"):
            s.module.start_command(s.state, str(path), s.start)
    assert s.module.start_command(s.state, str(write_command(tmp_path / "recorded_form.json", s.control)),
                                  s.start)[0][1:4] == ["-u", "-m", "mimo_bridge.train_mimo"]


def test_a_flag_named_twice_after_the_module_is_refused(tmp_path, monkeypatch):
    # finding F3 (probe R1): a second --out survived launch_command, the preflight allowed it (a run flag), and the
    # trainer's parser keeps the last value, so it would have written outside the folder its supervisor and watcher read
    s = setup(tmp_path, monkeypatch)
    elsewhere = str(tmp_path / "unwatched_folder")
    for name, extra, named in (
            ("out", ["--out", elsewhere], "--out"),
            ("storage", ["--checkpoint-storage", elsewhere + "/checkpoints", "--checkpoint-storage", elsewhere],
             "--checkpoint-storage"),
            ("resume", ["--resume", str(s.start)], "--resume"),
            ("switch", ["--fill-ring-first"], "--fill-ring-first"),
            ("two", ["--out", elsewhere, "--intention", "off"], "--intention, --out")):
        path = write_command(tmp_path / ("twice_%s.json" % name), s.control + extra)
        with pytest.raises(SystemExit, match="names %s more than once" % named):
            s.module.start_command(s.state, str(path), s.start)


def test_the_preflight_reads_a_byte_order_mark_file_as_the_relaunch_does(tmp_path, monkeypatch, capsys):
    # finding F4 (probe R3): training_preflight read --cmd-file as utf-8, and json refused the leading BOM that
    # PowerShell's Out-File writes, so `check --cmd-file` ended in a traceback where the relaunch read the same file
    s = setup(tmp_path, monkeypatch)
    path = write_command(tmp_path / "bom.json", s.control, bom=True)
    assert TP._command(SimpleNamespace(current=False, cmd_file=str(path), set=[])) == s.control
    assert s.module.start_command(s.state, str(path), s.start)[0] == s.control
    assert TP.main(["fingerprint", "--cmd-file", str(path)]) == 0
    printed = capsys.readouterr().out.strip()
    assert len(printed) == 16 and int(printed, 16) >= 0


def test_a_byte_order_mark_another_spelling_of_the_checkpoint_or_no_resume_is_accepted(tmp_path, monkeypatch):
    s = setup(tmp_path, monkeypatch)
    path = write_command(tmp_path / "bom.json", s.control, bom=True)             # PowerShell's Out-File writes one
    assert s.module.start_command(s.state, str(path), s.start)[0] == s.control
    spelled = TP.apply_sets(s.control, [("--resume", str(s.start).replace("\\", "/") + "/")])
    path = write_command(tmp_path / "spelled.json", spelled)
    assert s.module.start_command(s.state, str(path), s.start)[0] == spelled
    bare = list(s.control)
    del bare[bare.index("--resume"):bare.index("--resume") + 2]
    path = write_command(tmp_path / "no_resume.json", bare)
    cmd = s.module.launch_command(s.module.start_command(s.state, str(path), s.start)[0], s.start,
                                  s.module.run_folder("c"), [], s.preflight)[0]
    assert TP.flags_of(cmd)["--resume"] == str(s.start)
    assert TP.launch_refusal(cmd, root=s.root, src=s.src, registry=s.reg) is None
