# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE TRAINER CODE'S HASH (Source/Python/lem/trainer_code.py): one definition, read by the training preflight (a
record's code and code_hash, part of every launch command's fingerprint) and by the admission check of a replacement
hand calibration (lem.model.hand_response.trainer_code_hash). The independent review of branch
fix/hand-response-admission-2026-10-08 (D:/UEMA_Research/external/D/UEMA_scratch/plan_review/HAND_CALIBRATION_ADMISSION_REVIEW.md, section 3)
measured the old scope: production's checkout and a clean git archive of the same commit hashed differently (5 ignored
files of mimo_bridge/.pytest_cache), and a change under tokenizer/, which the trainer imports, left the hash unchanged.

Scope tested here: the .py files under lem/, mimo_bridge/ and tokenizer/ plus lem/master_config.yaml and every file
under mimo_bridge/assets/; without mimo_bridge/tools/, mimo_bridge/explainer/, mimo_bridge/experiments/, any tests
folder, any folder whose name starts with '.', and Python's caches; CRLF read as LF.
"""
from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from lem import trainer_code as TC
from lem.model import hand_response as HR
from mimo_bridge.tools import training_preflight as TP

REPO = Path(__file__).resolve().parents[4]
CMD = ["python", "-u", "-m", "mimo_bridge.train_mimo", "--resume", "C:/runs/a/checkpoints/ckpt_step_1", "--out",
       "C:/runs/a"]
# the 5 ignored files production's checkout held in mimo_bridge/.pytest_cache on Oct 8, 2026 (written Aug 23 to 26)
PYTEST_CACHE = {".gitignore": "# Created by pytest automatically.\n*\n",
                "CACHEDIR.TAG": "Signature: 8a477f597d28d172789f06886806bc55\n",
                "README.md": "# pytest cache directory #\n",
                "v/cache/lastfailed": "{}",
                "v/cache/nodeids": "[\n  \"mimo_bridge/tests/test_a.py::test_one\"\n]"}

# a stand-in trainer whose parser declares the flags of CMD (training_preflight.flag_problems reads them from its source)
TRAINER = ('import argparse\n\n\ndef parse_args(argv=None):\n    ap = argparse.ArgumentParser()\n'
           '    ap.add_argument("--resume")\n    ap.add_argument("--out")\n    return ap.parse_args(argv)\n')
IN_SCOPE = {"lem/__init__.py": "", "lem/model/core.py": "x = 1\n", "lem/master_config.yaml": "model:\n  width: 32\n",
            "mimo_bridge/__init__.py": "", "mimo_bridge/train_mimo.py": TRAINER,
            "mimo_bridge/assets/audio/bee_loop.wav": "RIFF-bee", "mimo_bridge/assets/audio/LICENSES.md": "public domain\n",
            "tokenizer/__init__.py": "", "tokenizer/config.py": "N = 3\n"}
OUT_OF_SCOPE = {"lem/notes.txt": "not code\n", "lem/tests/test_core.py": "t = 1\n",
                "mimo_bridge/phase0/benchmark_summary.json": "{}", "mimo_bridge/tools/dash.py": "z = 3\n",
                "mimo_bridge/tools/development_dashboard.js": "var a;\n", "mimo_bridge/explainer/schema.py": "s = 4\n",
                "mimo_bridge/explainer/viewer.js": "var v;\n", "mimo_bridge/experiments/e1_reach.py": "e = 5\n",
                "mimo_bridge/experiments/results/E1/args.json": "{}", "mimo_bridge/tests/test_a.py": "t = 6\n",
                "tokenizer/tests/test_audio.py": "t = 7\n", "lem/__pycache__/core.cpython-313.pyc": "\x00\x01",
                "lem/.cuda_build/kernel.py": "k = 8\n", "Source_other/x.py": "o = 9\n"}


def _write(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    return root


def _tree(tmp_path) -> Path:
    return _write(tmp_path / "src", dict(IN_SCOPE, **OUT_OF_SCOPE))


def _edit(path: Path, text="# changed\n"):
    path.write_bytes(path.read_bytes() + text.encode("utf-8"))


# ------------------------------------------------------------------------------------------------ the scope's rules
def test_the_scope_holds_the_trainers_python_files_and_data_files_only(tmp_path):
    src = _tree(tmp_path)
    assert set(TC.file_hashes(src)) == set(IN_SCOPE)
    assert all(TC.in_scope(rel) for rel in IN_SCOPE) and not any(TC.in_scope(rel) for rel in OUT_OF_SCOPE)


def test_an_ignored_pytest_cache_and_python_caches_leave_the_hash_unchanged(tmp_path):
    src = _tree(tmp_path)
    clean = TC.trainer_code_hash(src)
    fingerprint = TP.configuration(CMD, src)["fingerprint"]
    _write(src / "mimo_bridge" / ".pytest_cache", PYTEST_CACHE)
    _write(src / "mimo_bridge" / "tests" / ".pytest_cache", PYTEST_CACHE)
    _write(src, {"mimo_bridge/__pycache__/train_mimo.cpython-313.pyc": "\x00\x02", "tokenizer/.mypy_cache/x.py": "c"})
    assert TC.trainer_code_hash(src) == clean
    # the register's closing test of PREFLIGHT_FINGERPRINT_HASHES_PYTEST_CACHE: a pytest run that writes
    # mimo_bridge/.pytest_cache leaves every launch command's fingerprint unchanged
    assert TP.configuration(CMD, src)["fingerprint"] == fingerprint


def test_a_change_under_tokenizer_or_to_a_data_file_changes_the_hash(tmp_path):
    src = _tree(tmp_path)
    seen = {TC.trainer_code_hash(src)}
    for rel in ("tokenizer/config.py", "lem/master_config.yaml", "mimo_bridge/assets/audio/bee_loop.wav",
                "lem/model/core.py", "mimo_bridge/train_mimo.py"):
        _edit(src / rel)
        seen.add(TC.trainer_code_hash(src))
    assert len(seen) == 6                     # each of the 5 edits gave a new hash
    _write(src, {"tokenizer/modalities/touch.py": "u = 1\n"})            # a new file of the trainer code counts too
    assert TC.trainer_code_hash(src) not in seen


def test_a_change_under_the_explainer_the_tools_the_experiments_or_a_tests_folder_does_not(tmp_path):
    src = _tree(tmp_path)
    before = TC.trainer_code_hash(src)
    for rel in OUT_OF_SCOPE:
        _edit(src / rel)
    _write(src, {"mimo_bridge/explainer/capture.py": "c = 1\n", "mimo_bridge/tools/new_probe.py": "p = 1\n"})
    assert TC.trainer_code_hash(src) == before


def test_line_ends_are_read_as_lf(tmp_path):
    a = _write(tmp_path / "a", {"lem/model/core.py": "x = 1\ny = 2\n"})
    b = _write(tmp_path / "b", {"lem/model/core.py": "x = 1\r\ny = 2\r\n"})
    assert TC.trainer_code_hash(a) == TC.trainer_code_hash(b)


# ------------------------------------------------------------------------------------------------ one definition, two readers
def test_the_preflight_and_the_admission_check_read_the_same_function(tmp_path):
    src = _tree(tmp_path)
    assert HR.trainer_code_hash is TC.trainer_code_hash
    assert TP.code_hashes(src) == TC.file_hashes(src)
    conf = TP.configuration(CMD, src)
    assert conf["code"] == TC.file_hashes(src)
    assert conf["code_hash"] == TC.combined_hash(conf["code"]) == HR.trainer_code_hash(src)
    # what a preflight record stores about the code: the per-file hashes and their code hash, consistent
    path = TP.new_record(CMD, "the shared code hash", "tester", root=tmp_path / "pre", src=src)
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    assert rec["code"] == conf["code"] and rec["code_hash"] == TC.combined_hash(rec["code"]) == conf["code_hash"]
    # the default folder is the Source/Python this checkout's lem/trainer_code.py lies in
    assert TC.SOURCE_PYTHON == Path(TC.__file__).resolve().parents[1] == Path(HR.__file__).resolve().parents[2]


def _archive_head(dest: Path) -> Path:
    """A clean copy: git archive of this checkout's HEAD, Source/Python/lem, mimo_bridge and tokenizer only."""
    try:
        data = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(REPO), "archive", "--format=tar", "HEAD",
                               "Source/Python/lem", "Source/Python/mimo_bridge", "Source/Python/tokenizer"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip("git archive of HEAD is not readable here: %s" % exc)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")
    return dest / "Source" / "Python"


def test_a_checkout_holding_an_ignored_pytest_cache_hashes_as_a_clean_copy(tmp_path):
    """On the repository's own files: a clean git archive of HEAD, and the same files with production's 5 ignored
    .pytest_cache files added, give one code hash; a change under tokenizer/ gives another; changes under
    mimo_bridge/explainer/ and mimo_bridge/tools/ do not."""
    src = _archive_head(tmp_path / "head")
    clean = TC.trainer_code_hash(src)
    files = TC.file_hashes(src)
    assert any(k.startswith("tokenizer/") for k in files) and "lem/master_config.yaml" in files
    assert not any("/tests/" in k or k.startswith(TC.SKIP) for k in files)
    _write(src / "mimo_bridge" / ".pytest_cache", PYTEST_CACHE)
    assert TC.trainer_code_hash(src) == clean
    assert TP.configuration(CMD, src)["code_hash"] == clean
    _edit(src / "mimo_bridge" / "explainer" / "schema.py")
    _edit(src / "mimo_bridge" / "tools" / "training_preflight.py")
    assert TC.trainer_code_hash(src) == clean
    original = (src / "tokenizer" / "config.py").read_bytes()
    _edit(src / "tokenizer" / "config.py")
    assert TC.trainer_code_hash(src) != clean
    (src / "tokenizer" / "config.py").write_bytes(original)
    assert TC.trainer_code_hash(src) == clean
