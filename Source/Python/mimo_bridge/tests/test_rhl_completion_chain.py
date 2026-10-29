# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The continuation's completion chain (docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/completion_chain.py):
the trigger, the job conditions (the comparison's conditional second pass), the receipt and the status page, on fakes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

TOOL = (Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4" / "remembered_hand_learning_2026-10-11"
        / "completion_chain.py")
spec = importlib.util.spec_from_file_location("completion_chain", TOOL)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def jobs_file(tmp, page=None):
    spec = dict(end_step=489350, grace_s=0, status_page=str(page) if page else "", jobs=[
        dict(name="preserve", row="r_preserve", command=["py", "preserve.py", "{run}"], need_gb=2, when="always"),
        dict(name="reader", row="r_reader", command=["py", "reader.py", "--run", "{run}"], need_gb=4, when="ended",
             result=str(tmp / "reader.json")),
        dict(name="endpoint_1", row="r_end", command=["py", "cs.py", "evaluate"], need_gb=8, when="ended"),
        dict(name="endpoint_2", row="r_end2", command=["py", "cs.py", "evaluate", "--pass", "2"], need_gb=8,
             when=dict(file=str(tmp / "paired.json"), key="trigger.second_pass", equals=True))])
    p = tmp / "jobs.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def args(tmp, run):
    return SimpleNamespace(run=str(run), jobs=str(tmp / "jobs.json"), receipt=str(tmp / "receipt.json"),
                           trainer_pid=0, poll_s=0)


def fake_execute(tmp, second_pass):
    ran = []

    def execute(cmd):
        ran.append(cmd)
        if "reader.py" in cmd:
            (tmp / "reader.json").write_text(json.dumps(dict(verdict="pass")), encoding="utf-8")
        if "cs.py evaluate" in cmd and "--pass" not in cmd:
            (tmp / "paired.json").write_text(json.dumps(dict(trigger=dict(second_pass=second_pass))), encoding="utf-8")
        return 0
    return ran, execute


def test_ended_run_runs_every_job_and_the_conditional_pass_when_triggered(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=489350)), encoding="utf-8")
    page = tmp_path / "status.html"
    page.write_text("<p>a</p>\n%s\n<p>b</p>" % C.MARKER, encoding="utf-8")
    jobs_file(tmp_path, page)
    ran, execute = fake_execute(tmp_path, True)
    assert C.run_chain(args(tmp_path, run), sleep=lambda s: None, execute=execute) == 0
    r = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert r["trigger"] == "ended" and [e["runs"] for e in r["jobs"]] == [True, True, True, True]
    assert r["jobs"][1]["verdict"] == "pass" and str(run) in ran[1]
    assert "commit_guard --need-gb 4" in ran[1] and "test_runner run r_reader" in ran[1]
    assert r["status_page_updated"] and "completion chain ended" in page.read_text(encoding="utf-8")


def test_conditional_pass_skipped_when_not_triggered(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=489350)), encoding="utf-8")
    jobs_file(tmp_path)
    ran, execute = fake_execute(tmp_path, False)
    C.run_chain(args(tmp_path, run), sleep=lambda s: None, execute=execute)
    r = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert [e["runs"] for e in r["jobs"]] == [True, True, True, False] and len(ran) == 3


def test_early_end_runs_only_always_jobs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=487000)), encoding="utf-8")
    jobs_file(tmp_path)
    ran, execute = fake_execute(tmp_path, True)
    C.run_chain(args(tmp_path, run), sleep=lambda s: None, execute=execute)
    r = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert r["trigger"] == "ended_early" and [e["runs"] for e in r["jobs"]] == [True, False, False, False]


def test_trainer_gone_without_end_record_is_an_early_end_after_grace(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    jobs_file(tmp_path)
    ran, execute = fake_execute(tmp_path, True)
    C.run_chain(args(tmp_path, run), sleep=lambda s: None, execute=execute, alive=lambda: False)
    r = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert r["trigger"] == "ended_early" and r["stage_done"] is None and len(ran) == 1


def test_waits_while_the_trainer_lives(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    jobs_file(tmp_path)
    polls = []

    def sleep(s):
        polls.append(s)
        if len(polls) == 3:
            (run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=489350)), encoding="utf-8")
    ran, execute = fake_execute(tmp_path, False)
    C.run_chain(args(tmp_path, run), sleep=sleep, execute=execute, alive=lambda: True)
    assert len(polls) == 3 and json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))["trigger"] == "ended"


def test_malformed_jobs_are_refused(tmp_path):
    assert C.jobs_problems({}) != []
    bad = dict(end_step=1, jobs=[dict(name="a", row="r", command=[], need_gb=1, when="later")])
    probs = C.jobs_problems(bad)
    assert any("command" in p for p in probs) and any("when" in p for p in probs)
    (tmp_path / "jobs.json").write_text(json.dumps(bad), encoding="utf-8")
    assert C.run_chain(args(tmp_path, tmp_path), sleep=lambda s: None, execute=lambda c: 0) == 2


def test_fill_names_the_run_folder_and_its_name():
    assert C.fill("--run {run} --copy D:/x/{run_name}/y", Path("C:/a/H84_run_1")) ==         "--run %s --copy D:/x/H84_run_1/y" % Path("C:/a/H84_run_1")


def test_job_cwd_and_device_reach_the_runner_and_a_missing_folder_is_refused(tmp_path):
    job = dict(name="endpoint_1", row="r_end", command=["py", "cs.py"], need_gb=32, when="ended", cwd=str(tmp_path),
               device="gpu")
    cmd = C.job_command(job, tmp_path)
    assert "test_runner run r_end --device gpu --cwd %s -- py cs.py" % tmp_path in cmd
    plain = C.job_command(dict(job, cwd=None, device=None) and {k: v for k, v in job.items() if k not in ("cwd", "device")},
                          tmp_path)
    assert "--device" not in plain and "--cwd %s" % C.SOURCE in plain
    spec = dict(end_step=1, jobs=[dict(job, cwd=str(tmp_path / "missing")), dict(job, name="b", device="tpu")])
    probs = C.jobs_problems(spec)
    assert any("not a folder" in p for p in probs) and any("device" in p for p in probs)
    assert C.jobs_problems(dict(end_step=1, jobs=[job])) == []


def test_a_quoted_job_line_reaches_cmd_unchanged_through_its_batch_file(tmp_path):
    """The continuation's first chain (Oct 11, 2026, 6:33 PM PDT) passed each job line as one element of
    ["cmd.exe", "/c", line]; Python escaped its quotes and cmd.exe refused every job. A real quoted line now runs
    through run_in_batch: a quoted interpreter path, && and an exit code."""
    import sys
    marker = tmp_path / "ran.txt"
    py = sys.executable.replace("\\", "/")
    line = ('"%s" -c "import sys" && "%s" -c "open(r\'%s\', \'w\').write(\'ok\')"' % (py, py, str(marker)))
    assert C.run_in_batch(line, tmp_path / "jobs", "job_01") == 0
    assert marker.read_text() == "ok"
    assert C.run_in_batch('"%s" -c "import sys; sys.exit(3)"' % py, tmp_path / "jobs", "job_02") == 3
    import pytest
    with pytest.raises(ValueError):
        C.run_in_batch("echo %PATH%", tmp_path / "jobs", "job_03")
