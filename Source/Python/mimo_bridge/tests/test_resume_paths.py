# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""resume.py maps the saved command's development-repository inputs onto this checkout on every os.sep.

Before 2026-11-04 resume.py:57 compared each saved path with DEV_ROOT + os.sep: on macOS and Linux os.sep is "/",
the saved paths carry backslashes, so no input matched and lines 61-62 refused each one as an unmapped path of the
development machine. PathMapper compares both sides with backslashes replaced by "/" (apple port, REQ-A-8).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]              # Source/Python/mimo_bridge/tests -> the checkout


def _load_resume():
    """resume.py as a module (it sits at the checkout root, outside the mimo_bridge package)."""
    spec = importlib.util.spec_from_file_location("resume_script", REPO / "resume.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resume = _load_resume()
SAVED = json.loads((resume.BUNDLE / "run" / "command.json").read_text(encoding="utf-8"))
# the saved command's arguments after the interpreter (resume.py replaces the interpreter, saved[0], with its own)
DEV_INPUTS = [a for a in SAVED[1:] if a.lower().startswith(resume.DEV_ROOT.lower())]


@pytest.mark.parametrize("sep", ["/", "\\"])
def test_every_saved_development_input_maps_to_the_checkout(monkeypatch, tmp_path, sep):
    """With os.sep "/" (macOS, Linux) and "\\" (Windows): every C:\\...\\UEMA_devA_private input of the saved
    command maps to a file or folder of this checkout, and command() refuses none."""
    monkeypatch.setattr(os, "sep", sep)
    mapper = resume.PathMapper(resume.DEV_ROOT, REPO)
    assert len(DEV_INPUTS) == 6                                 # all 6 saved inputs (REQ-A-8's pass line)
    mapped = [mapper.local(a) for a in DEV_INPUTS]
    assert all(p is not None for p in mapped)
    assert all(p.exists() for p in mapped), [str(p) for p in mapped if not p.exists()]
    args = resume.command(tmp_path / "out", 100, mapper)      # SystemExit here = an input refused
    assert args[0] == sys.executable
    assert args[args.index("--max-steps") + 1] == str(resume.START_STEP + 100)
    assert args[args.index("--out") + 1] == str(tmp_path / "out" / "run")
    assert args[args.index("--resume") + 1] == str(resume.CHECKPOINT)
    # the saved "--device cuda:0" becomes the asked device, "auto" by default (the trainer resolves it), given once
    assert SAVED[SAVED.index("--device") + 1] == "cuda:0"
    assert args.count("--device") == 1 and args[args.index("--device") + 1] == "auto"
    cpu_args = resume.command(tmp_path / "out", 100, mapper, "cpu")
    assert cpu_args.count("--device") == 1 and cpu_args[cpu_args.index("--device") + 1] == "cpu"
    # the inputs keep their saved names (the trainer compares them with the saved ones; repo_paths re-roots them)
    for saved_input in DEV_INPUTS:
        assert saved_input in args


def test_saved_command_names_six_development_inputs():
    """The released command.json names 6 inputs inside the development repository (the caregiver speech library,
    three calibrations, the latent-body settings, the hand-choice record)."""
    assert len(DEV_INPUTS) == 6
    # the other drive-letter arguments are the 6 output paths command() replaces (--out, --checkpoint-storage,
    # --latent-body-output, --caregiver-record-dir, --resume, --latent-body-resume)
    assert len([a for a in SAVED[1:] if resume.PathMapper.foreign(a) and a not in DEV_INPUTS]) == 6


@pytest.mark.parametrize("path", [
    r"C:\Users\bhart\Documents\UEMA_devA_private\docs\a\b.json",
    "C:/Users/bhart/Documents/UEMA_devA_private/docs/a/b.json",
    r"c:\users\BHART\Documents\UEMA_devA_private\docs\a\b.json",
])
def test_path_mapper_normalises_slashes_and_case(path):
    """Backslashes, forward slashes and letter case all map to the same checkout path, keeping the case of the
    part after the development root."""
    mapper = resume.PathMapper(resume.DEV_ROOT, Path("/checkout"))
    assert mapper.local(path) == Path("/checkout") / "docs" / "a" / "b.json"


def test_path_mapper_leaves_other_paths_alone():
    """A path outside the development repository is not mapped; a drive-letter path is foreign; a relative or
    POSIX path is neither."""
    mapper = resume.PathMapper(resume.DEV_ROOT, Path("/checkout"))
    assert mapper.local(r"D:\UEMA_scratch\run") is None
    assert mapper.local(r"C:\Users\bhart\Documents\UEMA_devA_private_other\x") is None
    assert mapper.local(resume.DEV_ROOT) is None            # the root itself, no part after it
    assert mapper.foreign(r"D:\UEMA_scratch\run") is True
    assert mapper.foreign("D:/UEMA_scratch/run") is True
    assert mapper.foreign("/Users/someone/run") is False
    assert mapper.foreign("relative/run") is False
    assert mapper.foreign("0.05") is False


def test_command_refuses_missing_and_unmapped_inputs(monkeypatch, tmp_path):
    """An input the saved command names inside the development repository but absent from the checkout, and an
    input it names elsewhere on the development machine, are both refused by name; os.sep "/" as on macOS."""
    monkeypatch.setattr(os, "sep", "/")
    bundle = tmp_path / "bundle"
    (bundle / "run").mkdir(parents=True)
    (bundle / "run" / "command.json").write_text(json.dumps([
        "python.exe", "-u", "-m", "mimo_bridge.train_mimo",
        "--a", resume.DEV_ROOT + r"\docs\absent_file.json",
        "--b", r"D:\UEMA_other\thing.json",
        "--c", "0.5",
        "--max-steps", "494025",
    ]), encoding="utf-8")
    monkeypatch.setattr(resume, "BUNDLE", bundle)
    with pytest.raises(SystemExit) as exc:
        resume.command(tmp_path / "out", 100, resume.PathMapper(resume.DEV_ROOT, tmp_path / "checkout"))
    text = str(exc.value)
    assert "--a" in text and "not in this repository" in text
    assert "--b" in text and "does not map" in text
    assert "--c" not in text
