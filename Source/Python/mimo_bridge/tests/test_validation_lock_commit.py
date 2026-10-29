# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The operator's approval commits the locked set alone (validation_lock._commit), checked in a
throwaway repository: the subject names the approver, the commit holds only the three locked files,
another session's uncommitted work stays out of it, and the private index is removed afterwards.
(test_validation_lock.py is itself protected by the guard; this file adds the commit check beside it.)"""
import subprocess

from mimo_bridge.tools.observer import validation_lock as VL


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True).stdout.strip()


def test_the_approval_commits_only_the_locked_set(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "README").write_text("x\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "start")
    (repo / "README").write_text("another session's change, not committed\n")
    locked = repo / "docs" / "locked"
    exp = dict(cases=dict(a=dict(present=["H05"], absent=[], truth={}, qpos_hash="q1")), known_defects={}, boundary={})
    golden = dict(detector_version="v", cases=dict(a=dict(qpos_hash="q1", detections=[])))
    VL.init_from(exp, golden, by="test", source="test", folder=locked)
    monkeypatch.setattr(VL, "ROOT", repo)
    monkeypatch.setattr(VL, "LOCKED", locked)
    monkeypatch.setattr(VL, "_interactive", lambda: True)
    pid = "current_" + VL.verify(locked)["lock"]["files"]["expectations.json"][:8]
    answers = iter([pid, "Brian"])
    out = VL.approve(current=True, commit=True, ask=lambda prompt: next(answers))
    assert out["commit"] == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "log", "-1", "--format=%s") == "Operator Brian approves observer validation's pass/fail lines (%s)" % pid
    assert sorted(_git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()) == [
        "docs/locked/expectations.json", "docs/locked/golden.json", "docs/locked/lock.json"]
    assert VL.uncommitted(locked) == []
    assert _git(repo, "status", "--porcelain", "--", "README") == "M README", "another session's work was touched"
    assert not (locked.parent / ".lock_index").exists()
    assert VL.verify(locked)["lock"]["approved_by"] == "Brian"
