# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The dashboard's three macOS lines (docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html, REQ-A-8 c; the plan's row
WI-9): the Live tab's copy-and-paste text and the interpreter the 3D viewer button starts come from one table keyed by
sys.platform (development_dashboard.VIEWER_LAUNCH; on macOS mujoco.viewer.launch_passive runs only under mjpython), the
movement viewer's pictures folder and recording roots carry no drive letter (movement_viewer.pictures_dir,
sources.archive_root), and commit_guard.HEAVY_LOCK lies under the checkout. Under sys.platform "win32" every line reads
as it did before the change: the PowerShell text, .venv_mimo\\Scripts\\python.exe first, D:/UEMA_archive beside a
checkout on D:.

Every test patches sys.platform and uses its own folders; none starts a process (subprocess.Popen is replaced) and none
reads the real lock or pictures folder.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mimo_bridge.tools import commit_guard
from mimo_bridge.tools import development_dashboard as dashboard
from mimo_bridge.tools import movement_viewer as viewer
from mimo_bridge.tools.movement_viewer import sources

RUN = "Saved/mimo_bridge/experiments/H84_test_seg1"
# the Windows text as the dashboard showed it before this change (development_dashboard.py:1501-1502 on Nov 4, 2026)
WINDOWS_TEXT = ('$env:PYTHONPATH="Source\\Python"; .venv_mimo\\Scripts\\python.exe -m mimo_bridge.tools.live_viewer '
                '--run "' + RUN + '"')


class _Proc:
    pid = 4242


def _fake_popen(calls):
    def popen(args, **kw):
        calls.append((list(args), kw))
        return _Proc()
    return popen


# ---- the Live tab's copy-and-paste text ---------------------------------------------------------------------------

def test_under_darwin_the_live_tab_text_is_a_shell_line_naming_mjpython(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    text = dashboard.live_state(tmp_path, {"run": RUN}, row={})["viewer_command"]
    assert text.startswith("PYTHONPATH=Source/Python"), text
    assert "mjpython" in text and "-m mimo_bridge.tools.live_viewer" in text, text
    assert text.endswith('--run "%s"' % RUN) and "$env:" not in text and "\\" not in text, text


def test_under_win32_the_live_tab_text_reads_as_today(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    assert dashboard.live_state(tmp_path, {"run": RUN}, row={})["viewer_command"] == WINDOWS_TEXT


def test_under_linux_and_an_unlisted_platform_the_text_is_the_sh_line_with_the_venvs_python(monkeypatch):
    for platform in ("linux", "freebsd14"):
        row = dashboard.viewer_launch(platform)
        text = row["shell"].format(run=RUN)
        assert text.startswith("PYTHONPATH=Source/Python .venv/bin/python -m mimo_bridge.tools.live_viewer"), text
        assert "mjpython" not in text
    assert dashboard.viewer_launch("darwin") is dashboard.VIEWER_LAUNCH["darwin"]
    monkeypatch.setattr(sys, "platform", "darwin")
    assert dashboard.viewer_launch() is dashboard.VIEWER_LAUNCH["darwin"]


# ---- the interpreter the 3D viewer button starts ---------------------------------------------------------------

def test_under_darwin_the_viewer_button_starts_the_venvs_mjpython(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    exe = tmp_path / ".venv" / "bin" / "mjpython"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(calls))
    out = dashboard.live_command(tmp_path, {"run": RUN}, {"action": "open_viewer"})
    assert out == {"opened": True, "pid": 4242}
    args, kw = calls[0]
    assert args == [str(exe), "-u", "-m", "mimo_bridge.tools.live_viewer", "--run", str(tmp_path / RUN)]
    assert kw["env"]["PYTHONPATH"] == str(tmp_path / "Source" / "Python") and kw["cwd"] == str(tmp_path)
    assert (tmp_path / RUN / "live" / "viewer.log").exists()


def test_under_darwin_without_a_venv_the_button_takes_mjpython_beside_the_running_python_else_refuses(monkeypatch,
                                                                                                        tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    running = tmp_path / "env" / "bin" / "python3"
    running.parent.mkdir(parents=True)
    running.write_text("")
    monkeypatch.setattr(sys, "executable", str(running))
    # no mjpython anywhere: refused with the three places looked (the route answers 400 with this text), never the
    # running python, under which launch_passive refuses in viewer.log while the page says the window opened
    with pytest.raises(ValueError) as refused:
        dashboard.viewer_interpreter(tmp_path)
    text = str(refused.value)
    for place in (tmp_path / ".venv" / "bin" / "mjpython", tmp_path / ".venv_mimo" / "bin" / "mjpython",
                  running.with_name("mjpython")):
        assert str(place) in text, (place, text)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(calls))
    with pytest.raises(ValueError):
        dashboard.live_command(tmp_path, {"run": RUN}, {"action": "open_viewer"})
    assert calls == [], "nothing is started when no mjpython exists"
    beside = running.with_name("mjpython")
    beside.write_text("")
    assert dashboard.viewer_interpreter(tmp_path) == beside
    venv = tmp_path / ".venv_mimo" / "bin" / "mjpython"
    venv.parent.mkdir(parents=True)
    venv.write_text("")
    assert dashboard.viewer_interpreter(tmp_path) == venv, "a checkout venv's mjpython comes first"


def test_under_win32_the_viewer_button_starts_the_venvs_python_exe_as_today(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    assert dashboard.viewer_interpreter(tmp_path) == Path(sys.executable), "no .venv_mimo: the running python, as today"
    exe = tmp_path / ".venv_mimo" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert dashboard.viewer_interpreter(tmp_path) == exe
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(calls))
    dashboard.live_command(tmp_path, {"run": RUN}, {"action": "open_viewer"})
    assert calls[0][0][0] == str(exe) and "mjpython" not in " ".join(calls[0][0])


# ---- the movement viewer's pictures folder ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_pictures_dir_is_the_checkout_folder_when_it_exists_else_beside_the_index(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)
    checkout = tmp_path / "external" / "D" / "UEMA_movement_viewer" / "pictures"
    monkeypatch.setattr(viewer, "PICTURES_DEFAULT", checkout)
    monkeypatch.setattr(viewer, "HOME", tmp_path / "Saved" / "mimo_bridge" / "movement_viewer")
    monkeypatch.delenv(viewer.PICTURES_ENV, raising=False)
    assert viewer.pictures_dir() == tmp_path / "Saved" / "mimo_bridge" / "movement_viewer" / "pictures"
    checkout.mkdir(parents=True)
    assert viewer.pictures_dir() == checkout
    assert viewer.pictures_dir(env={viewer.PICTURES_ENV: str(tmp_path / "elsewhere")}) == tmp_path / "elsewhere"
    monkeypatch.setenv(viewer.PICTURES_ENV, str(tmp_path / "named"))
    assert viewer.pictures_dir() == tmp_path / "named"


def test_the_pictures_default_lies_under_the_checkout_with_no_drive_letter_of_its_own():
    checkout = Path(viewer.__file__).resolve().parents[5]
    assert viewer.PICTURES_DEFAULT == checkout / "external" / "D" / "UEMA_movement_viewer" / "pictures"
    assert viewer.REPO == checkout


# ---- the recordings' roots ----------------------------------------------------------------------------------------

def test_every_recording_root_is_under_the_checkout_or_the_archive_root():
    checkout = str(sources.REPO)
    archive = sources.archive_root()
    for where, pattern in sources.ROOTS:
        assert pattern.startswith(checkout) or pattern.startswith(archive), (where, pattern)
    assert [w for w, _ in sources.ROOTS].count("archive") == 2


def test_archive_root_reads_the_named_folder_then_the_mirror_then_on_windows_the_checkouts_drive(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "REPO", tmp_path / "checkout")
    (tmp_path / "checkout").mkdir()
    mirror = tmp_path / "checkout" / "external" / "D" / "UEMA_archive"
    assert sources.archive_root("darwin", env={}) == str(mirror), "absent on macOS: a folder whose patterns match nothing"
    assert sources.archive_root("win32", env={}) == str(Path((tmp_path / "checkout").anchor) / "UEMA_archive"), \
        "as today on Windows: UEMA_archive at the root of the checkout's drive"
    mirror.mkdir(parents=True)
    for platform in ("darwin", "win32", "linux"):
        assert sources.archive_root(platform, env={}) == str(mirror)
    assert sources.archive_root("darwin", env={sources.ARCHIVE_ENV: str(tmp_path / "named")}) == str(tmp_path / "named")
    monkeypatch.setenv(sources.ARCHIVE_ENV, str(tmp_path / "from_env"))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert sources.archive_root() == str(tmp_path / "from_env")


# ---- the heavy-job lock ----------------------------------------------------------------------------------------------

def test_the_heavy_job_lock_lies_under_the_checkout():
    checkout = Path(commit_guard.__file__).resolve().parents[4]
    assert commit_guard.REPO == checkout
    assert Path(commit_guard.HEAVY_LOCK) == checkout / "Saved" / "gof_observer_20260924" / "heavy_job.lock"
    assert "bhart" not in commit_guard.HEAVY_LOCK.replace(str(checkout), "")
    assert commit_guard.heavy_job_extra_gb.__defaults__[0] == commit_guard.HEAVY_LOCK


def test_no_tool_file_of_this_item_names_a_drive_letter_path():
    """REQ-A-8 (c)'s pass line, 'grep for [CD]:[\\\\/] in the tool files returns 0 lines': the four tool files of this item
    carry no C:/ or D:/ path on any line, comments and docstrings included (conftest.py, the fifth file of that line, is
    another item's)."""
    import re
    for module in (dashboard, commit_guard, viewer, sources):
        lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
        hits = [line for line in lines if re.search(r"[CD]:[\\/]", line)]
        assert hits == [], (module.__name__, hits)
