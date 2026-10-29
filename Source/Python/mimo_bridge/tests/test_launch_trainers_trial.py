# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The relaunch's running-trainer refusal counts trial trainers (the continuation block's review of Oct 11, 2026, finding
B2): a common-start pass trains inside common_starts_trial_trainer.py, whose command line never names
mimo_bridge.train_mimo, so launch_h84_seg41_2026-09-29.trainers() used to miss it and only the memory guard stood between
a relaunch and a running pass. psutil.process_iter is replaced by a list of command lines written as text: no process is
listed, started or ended."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs/gain_of_function_v4/coordinator/launch_h84_seg41_2026-09-29.py"
PY = r"C:\Users\bhart\Documents\UEMA_devA_private\.venv_mimo\Scripts\python.exe"
ARM = r"C:\Users\bhart\Documents\UEMA_wt\rhl_compare\docs\gain_of_function_v4\remembered_hand_learning_2026-10-11"


def launch():
    spec = importlib.util.spec_from_file_location("launch_trainers_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_iter(cmdlines):
    procs = [SimpleNamespace(pid=100 + i, info=dict(pid=100 + i, name=name, cmdline=c))
             for i, (name, c) in enumerate(cmdlines)]
    return lambda attrs=None: iter(procs)


def test_a_trial_trainer_and_train_mimo_count_and_nothing_else_does(monkeypatch):
    import psutil
    L = launch()
    lines = [
        ("python.exe", [PY, "-u", "-m", "mimo_bridge.train_mimo", "--resume", "x"]),                 # production trainer
        ("python.exe", [PY, "-B", ARM + r"\common_starts_trial_trainer.py", "--out", "arm_0.json"]),  # a pass's arm
        ("python.exe", [PY, "-m", "pytest", "mimo_bridge/tests/test_rhl_common_starts.py"]),         # a test run
        ("python.exe", [PY, "-B", r"C:\x\tests\test_something_trial_trainer.py"]),                   # a test file
        ("python.exe", [PY, "-u", r"C:\x\hourly_watch.py", "run", "8760"]),                           # a watcher
        ("python.exe", [PY, "-B", ARM + r"\common_starts.py", "evaluate", "--pass-label", "baseline_1"]),  # the driver
        ("node.exe", ["node", "x_trial_trainer.py"]),                                                 # not python
    ]
    monkeypatch.setattr(psutil, "process_iter", fake_iter(lines))
    assert [p.pid for p in L.trainers()] == [100, 101]


def test_script_of_skips_the_interpreters_own_options():
    L = launch()
    assert L._script_of([PY, "-X", "utf8", "-B", "a_trial_trainer.py", "-m", "x"]) == "a_trial_trainer.py"
    assert L._script_of([PY, "-m", "mimo_bridge.train_mimo"]) is None
    assert L._script_of([PY]) is None
