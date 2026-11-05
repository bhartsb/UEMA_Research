# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Mode B's Windows named shared memory refuses to open off Windows, by name.

Mode B (shm_writer.py, lem/action/shm.py, tokenizer/transport/cpu_reader.py) is the Unreal
transport and is off the training path (datasource.py:3-8). Off Windows, every open in
shm_writer.py (PipelineSHMWriter, VisionSHMWriter, ActionSHMReader) and lem/action/shm.py
(ActionSHM.open) raises RuntimeError whose message contains "Windows named shared memory",
the platform string and the region name, instead of a bare mmap error or a silent False.
cpu_reader.py's CPUReader.open has no guard (its mmap failure is logged and returns False;
the file is not in this work item), so it gets only the import check: the three modules
import without calling mmap.mmap, checked in a child interpreter with mmap.mmap replaced by
a function that raises. The guard reads sys.platform at call time unless a platform string
is passed, so the tests use both: monkeypatch sets sys.platform to "darwin", and one test
passes "darwin" and "linux" explicitly.
"""
import os
import subprocess
import sys

import pytest

from lem.action import shm as ASHM
from lem.action.writer import ACTION_FRAME_SIZE
from mimo_bridge.shm_writer import ActionSHMReader, PipelineSHMWriter, VisionSHMWriter

SRC_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Cap on the child interpreter of test_modules_import_without_calling_mmap, in seconds. The
# child imports numpy only (tokenizer/config.py and lem/action/writer.py import torch under
# TYPE_CHECKING alone); the whole test measured 3.7 s on the development machine, 2026-11-04.
CHILD_TIMEOUT_S = 300


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


def test_guard_names_platform_and_region(darwin):
    with pytest.raises(RuntimeError, match="Windows named shared memory") as info:
        ASHM.require_windows_named_shared_memory("LEM_Pipeline_guard_test")
    assert "darwin" in str(info.value) and "LEM_Pipeline_guard_test" in str(info.value)


def test_guard_judges_the_platform_argument_when_given():
    """The platform string is injectable: "darwin" and "linux" raise, "win32" passes, with
    sys.platform untouched."""
    for platform in ("darwin", "linux"):
        with pytest.raises(RuntimeError, match="Windows named shared memory") as info:
            ASHM.require_windows_named_shared_memory("LEM_Vision_guard_test", platform=platform)
        assert platform in str(info.value) and "LEM_Vision_guard_test" in str(info.value)
    assert ASHM.require_windows_named_shared_memory("LEM_Vision_guard_test", platform="win32") is None


def test_action_shm_open_raises_off_windows(darwin):
    shm = ASHM.ActionSHM("LEM_Action_guard_test", ACTION_FRAME_SIZE)
    with pytest.raises(RuntimeError, match="Windows named shared memory"):
        shm.open()          # raises; does not return False as the Windows failure branch does
    assert shm.buf is None
    shm.close()             # closing an unopened region is allowed


def test_pipeline_vision_and_action_reader_raise_off_windows(darwin):
    with pytest.raises(RuntimeError, match="Windows named shared memory") as info:
        PipelineSHMWriter(name="LEM_Pipeline_guard_test")
    assert "LEM_Pipeline_guard_test" in str(info.value)
    with pytest.raises(RuntimeError, match="Windows named shared memory") as info:
        VisionSHMWriter(name="LEM_Vision_guard_test")
    assert "LEM_Vision_guard_test" in str(info.value)
    with pytest.raises(RuntimeError, match="Windows named shared memory") as info:
        ActionSHMReader(name="LEM_Action_guard_test", retries=1)
    assert "LEM_Action_guard_test" in str(info.value)


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="the Windows branch of the guard")
def test_guard_passes_on_windows():
    assert ASHM.require_windows_named_shared_memory("LEM_Pipeline_guard_test") is None


def test_modules_import_without_calling_mmap():
    """Import the three Mode B modules in a child interpreter whose mmap.mmap raises."""
    code = (
        "import mmap\n"
        "def _refuse(*a, **k): raise AssertionError('mmap.mmap called at import')\n"
        "mmap.mmap = _refuse\n"
        "import lem.action.shm, mimo_bridge.shm_writer, tokenizer.transport.cpu_reader\n"
        "print('imported without mmap')\n"
    )
    env = dict(os.environ, PYTHONPATH=SRC_PY)
    out = subprocess.run([sys.executable, "-c", code], cwd=SRC_PY, env=env,
                         capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    assert out.returncode == 0, out.stderr
    assert "imported without mmap" in out.stdout
