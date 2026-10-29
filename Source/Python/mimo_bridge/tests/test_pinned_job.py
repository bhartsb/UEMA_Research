# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/pinned_job.py: a pause job runs the code of one commit, never a worktree's working files
(review finding 4 of docs/gain_of_function_v4/agents/proprio_token_review_2026-09-25.md).

Pinned here, on a small repository built in pytest's temporary folder: the job runs the committed file even when the
working copy differs; the export is checked file by file and deleted after; the runner refuses when its own copy is
not the commit's, when a required receipt is for another commit or failed, and when the export holds a file too many
or a changed one; the job's exit code comes back and lands in the receipt; an export left by a run ended from outside
is removed by the next run, one whose runner still runs is kept."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4" / "pinned_job.py"

JOB = '''import json, os, sys
from pathlib import Path
import mimo_bridge
Path(sys.argv[1]).write_text(json.dumps(dict(version="{version}", file=__file__, cwd=os.getcwd(),
    commit=os.environ.get("PINNED_COMMIT"), root=os.environ.get("PINNED_ROOT"), bridge=mimo_bridge.__file__)))
sys.exit(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
'''


def run_git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    # a repository on D: is "dubious ownership" to git; accept it for this test's git commands only (as production's
    # test settings do), never in the user's git settings
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "*")
    for k, v in (("GIT_AUTHOR_NAME", "test"), ("GIT_AUTHOR_EMAIL", "test@example.com"),
                 ("GIT_COMMITTER_NAME", "test"), ("GIT_COMMITTER_EMAIL", "test@example.com")):
        monkeypatch.setenv(k, v)
    r = tmp_path / "repo"
    (r / "docs" / "gain_of_function_v4").mkdir(parents=True)
    (r / "Source" / "Python" / "mimo_bridge").mkdir(parents=True)
    shutil.copyfile(RUNNER, r / "docs" / "gain_of_function_v4" / "pinned_job.py")
    (r / "Source" / "Python" / "mimo_bridge" / "__init__.py").write_text("# stub\n", encoding="utf-8")
    (r / "Source" / "Python" / "job.py").write_text(JOB.replace("{version}", "committed"), encoding="utf-8")
    (r / "Source" / "Python" / "data.json").write_text('{"a": 1}\n', encoding="utf-8")
    run_git(r, "init", "-q")
    run_git(r, "add", "docs", "Source")
    run_git(r, "commit", "-q", "-m", "one")
    return r, run_git(r, "rev-parse", "HEAD")


def pin(repo, commit, scratch, *extra, out, code=0):
    runner = repo / "docs" / "gain_of_function_v4" / "pinned_job.py"
    cmd = [sys.executable, str(runner), "--commit", commit[:10], "--path", "Source/Python", "--path",
           "docs/gain_of_function_v4", "--scratch", str(scratch), *extra, "--",
           sys.executable, "{root}/Source/Python/job.py", str(out), str(code)]
    return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=""))


def test_the_job_runs_the_committed_code_from_a_checked_export_that_is_deleted_after(repo, tmp_path):
    r, commit = repo
    # the working copy moves on after the commit: the job must not see it
    (r / "Source" / "Python" / "job.py").write_text(JOB.replace("{version}", "working copy"), encoding="utf-8")
    out, scratch = tmp_path / "out.json", tmp_path / "scratch"
    p = pin(r, commit, scratch, "--receipt", str(tmp_path / "receipt.json"), out=out)
    assert p.returncode == 0, p.stdout + p.stderr
    got = json.loads(out.read_text())
    assert got["version"] == "committed" and got["commit"] == commit
    root = Path(got["root"])
    assert root.parent == scratch and root.name.startswith("pinned_" + commit[:12])
    for key in ("file", "bridge", "cwd"):
        assert Path(got[key]).resolve().is_relative_to(root.resolve()), (key, got[key])
    assert not root.exists(), "the export is deleted after the job"
    assert "checked against the commit" in p.stdout
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["commit"] == commit and receipt["exit_code"] == 0


def test_the_jobs_exit_code_comes_back_and_a_required_receipt_must_be_the_same_commit_and_passed(repo, tmp_path):
    r, commit = repo
    out, scratch = tmp_path / "out.json", tmp_path / "scratch"
    failed = tmp_path / "failed.json"
    p = pin(r, commit, scratch, "--receipt", str(failed), out=out, code=7)
    assert p.returncode == 7 and json.loads(failed.read_text())["exit_code"] == 7
    out.unlink()
    p = pin(r, commit, scratch, "--require", str(failed), out=out)
    assert p.returncode == 3 and not out.exists() and "exit code 7" in p.stdout
    other = tmp_path / "other.json"
    other.write_text(json.dumps(dict(commit="0" * 40, exit_code=0)))
    p = pin(r, commit, scratch, "--require", str(other), out=out)
    assert p.returncode == 3 and not out.exists()
    good = tmp_path / "good.json"
    good.write_text(json.dumps(dict(commit=commit, exit_code=0)))
    assert pin(r, commit, scratch, "--require", str(good), out=out).returncode == 0 and out.exists()


def test_a_command_that_cannot_start_is_refused_and_its_export_deleted(repo, tmp_path):
    r, commit = repo
    runner = r / "docs" / "gain_of_function_v4" / "pinned_job.py"
    scratch = tmp_path / "scratch"
    p = subprocess.run([sys.executable, str(runner), "--commit", commit, "--path", "Source/Python", "--path",
                        "docs/gain_of_function_v4", "--scratch", str(scratch), "--receipt", str(tmp_path / "r.json"),
                        "--", str(tmp_path / "no_such_program.exe")], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=""))
    assert p.returncode == 3 and "could not start" in p.stdout, p.stdout + p.stderr
    assert json.loads((tmp_path / "r.json").read_text())["exit_code"] == 3
    assert not list(scratch.glob("pinned_*"))


def test_the_job_ends_when_the_process_that_launched_it_ends(repo, tmp_path):
    """The pause runner's timeout kills only the watchdog it started; the pinned job must not outlive it."""
    import psutil
    r, commit = repo
    runner = r / "docs" / "gain_of_function_v4" / "pinned_job.py"
    pid_file = tmp_path / "job_pid.txt"
    job = ("import os, sys, time, pathlib; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
           "time.sleep(120)")
    inner = [sys.executable, str(runner), "--commit", commit, "--path", "Source/Python", "--path",
             "docs/gain_of_function_v4", "--scratch", str(tmp_path / "scratch"), "--poll", "0.5",
             "--receipt", str(tmp_path / "r.json"), "--", sys.executable, "-c", job, str(pid_file)]
    launcher = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; "
                                 "subprocess.Popen(sys.argv[1:]); time.sleep(120)"] + inner,
                                env=dict(os.environ, PYTHONPATH=""))
    deadline = time.time() + 90
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.5)
    assert pid_file.exists(), "the job never started"
    job_pid = int(pid_file.read_text())
    launcher.kill()                                   # what the runner's timeout does to the watchdog
    launcher.wait()
    deadline = time.time() + 30
    while psutil.pid_exists(job_pid) and time.time() < deadline:
        time.sleep(0.5)
    assert not psutil.pid_exists(job_pid), "the job outlived the process that launched it"
    deadline = time.time() + 30
    while not (tmp_path / "r.json").exists() and time.time() < deadline:
        time.sleep(0.5)
    assert json.loads((tmp_path / "r.json").read_text())["exit_code"] == 6
    assert not list((tmp_path / "scratch").glob("pinned_*"))


def test_an_edited_runner_refuses_and_runs_nothing(repo, tmp_path):
    r, commit = repo
    runner = r / "docs" / "gain_of_function_v4" / "pinned_job.py"
    runner.write_text(runner.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    out = tmp_path / "out.json"
    p = pin(r, commit, tmp_path / "scratch", out=out)
    assert p.returncode == 3 and not out.exists() and "differs from commit" in p.stdout
    assert not list((tmp_path / "scratch").glob("pinned_*"))


def test_the_check_refuses_an_extra_or_changed_file_and_accepts_gits_line_ends(repo, tmp_path):
    spec = importlib.util.spec_from_file_location("pinned_job_under_test", RUNNER)
    J = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(J)
    r, commit = repo
    expected = J.tree(r, commit, ("Source/Python",))
    dest = tmp_path / "export"
    J.export(r, commit, ("Source/Python",), dest)
    counts = J.verify(dest, expected)
    assert counts["exact"] + counts["line ends"] == len(expected) == 3
    crlf = dest / "Source" / "Python" / "data.json"
    crlf.write_bytes(crlf.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert J.verify(dest, expected)["line ends"] >= 1, "git's own CRLF export is the same file"
    (dest / "Source" / "Python" / "extra.py").write_text("x = 1\n")
    with pytest.raises(J.PinError, match="1 extra"):
        J.verify(dest, expected)
    (dest / "Source" / "Python" / "extra.py").unlink()
    crlf.write_text('{"a": 2}\n', encoding="utf-8")
    with pytest.raises(J.PinError, match="not the commit's blob"):
        J.verify(dest, expected)


def test_an_export_left_by_an_ended_run_is_removed_and_a_running_ones_is_kept(repo, tmp_path):
    import psutil
    r, commit = repo
    scratch = tmp_path / "scratch"
    dead, live, unknown = scratch / "pinned_dead", scratch / "pinned_live", scratch / "pinned_unknown"
    for f in (dead, live, unknown):
        f.mkdir(parents=True)
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    (dead / ".pinned_job_process.json").write_text(json.dumps(dict(pid=ended.pid, created=12345.0)))
    me = psutil.Process(os.getpid())
    (live / ".pinned_job_process.json").write_text(json.dumps(dict(pid=me.pid, created=me.create_time())))
    p = pin(r, commit, scratch, out=tmp_path / "out.json")
    assert p.returncode == 0, p.stdout + p.stderr
    assert not dead.exists() and live.exists() and unknown.exists(), "only a proven-ended run's export is removed"


def test_a_probe_run_in_the_export_records_the_pinned_commit(tmp_path, monkeypatch):
    """The probe harness reads its code's commit from git, and an export holds no .git: a probe run there recorded no
    commit, so its failures never counted as this code's and were run again at every pass (found 2026-09-26 by
    test_suite_records_a_failing_probe_and_runs_the_rest in the pinned unit tests). In the export named by
    PINNED_ROOT it records PINNED_COMMIT, with no file differing; anywhere else the variables change nothing."""
    from mimo_bridge.tools import gof_probe_harness as H
    export = tmp_path / "pinned_export"
    export.mkdir()
    saved = dict(H._CODE)
    monkeypatch.setattr(H, "REPO", export)
    try:
        monkeypatch.delenv("PINNED_COMMIT", raising=False)
        monkeypatch.delenv("PINNED_ROOT", raising=False)
        assert H.pinned_commit() is None
        assert H.refresh_code_commit()["commit"] is None, "no git and no pin: no commit"
        monkeypatch.setenv("PINNED_COMMIT", "c0ffee")
        monkeypatch.setenv("PINNED_ROOT", str(tmp_path / "another_export"))
        assert H.pinned_commit() is None, "another folder's pin is not this code's"
        monkeypatch.setenv("PINNED_ROOT", str(export).upper() if os.name == "nt" else str(export))
        assert H.pinned_commit() == "c0ffee"
        code = H.refresh_code_commit()
        assert code["commit"] == "c0ffee" and code["uncommitted_tracked_files"] == 0 and code["pinned_export"]
    finally:
        H._CODE.clear()
        H._CODE.update(saved)
