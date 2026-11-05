# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The free memory guard exits 0 at or above --need-gb GB of available physical memory and 1 under it.

Every exit-code test injects the memory reading, so the result does not depend on this machine's
memory at test time; one test reads psutil for real and checks only that the number is an int at or
above 0; one runs the module as a command with a 0 GB line, which holds on any machine.
"""
import io
import pathlib
import subprocess
import sys

import pytest

from mimo_bridge.tools import free_memory_guard as guard

GB = 1024 ** 3
# The WI-1 acceptance run's own output line on November 4, 2026 (UEMA_APPLE_PORT_IMPLEMENTATION_PLAN
# section 9): "available 8,366,292,992 bytes (7.79 GB of 1,073,741,824 bytes); need 12.00 GB ... refused".
ACCEPTANCE_RUN_AVAILABLE_BYTES = 8_366_292_992
SOURCE_PYTHON_ROOT = pathlib.Path(guard.__file__).resolve().parents[2]  # .../Source/Python


def test_one_gb_is_2_to_the_30_bytes():
    assert guard.BYTES_PER_GB == 1_073_741_824


@pytest.mark.parametrize("need_gb, available, allowed", [
    (12, 12 * GB, True),          # exactly on the line holds
    (12, 12 * GB - 1, False),     # one byte under refuses
    (12, ACCEPTANCE_RUN_AVAILABLE_BYTES, False),   # 7.79 GB, the acceptance run's reading, refused
    (0, 0, True),                 # a zero line always holds
    (1.5, 1_610_612_736, True),   # 1.5 GB = 1,610,612,736 bytes
    (1.5, 1_610_612_735, False),
])
def test_verdict_compares_bytes_against_need_gb(need_gb, available, allowed):
    v = guard.verdict(need_gb, available)
    assert v.allowed is allowed
    assert v.exit_code == (0 if allowed else 1)
    assert v.need_bytes == int(need_gb * GB)
    assert v.available_bytes == available


def test_verdict_rejects_negative_or_infinite_lines():
    with pytest.raises(ValueError):
        guard.verdict(-1, 10 * GB)
    with pytest.raises(ValueError):
        guard.verdict(float("nan"), 10 * GB)
    with pytest.raises(ValueError):
        guard.verdict(float("inf"), 10 * GB)
    with pytest.raises(ValueError):
        guard.verdict(1, -5)


def test_main_exit_0_when_the_line_holds():
    out = io.StringIO()
    code = guard.main(["--need-gb", "12"], reader=lambda: 20 * GB, out=out)
    assert code == 0
    text = out.getvalue()
    assert "available 21,474,836,480 bytes (20.00 GB" in text
    assert "need 12.00 GB (12,884,901,888 bytes)" in text
    assert text.strip().endswith("allowed (exit 0)")


def test_main_exit_1_when_memory_is_under_the_line():
    out = io.StringIO()
    code = guard.main(["--need-gb", "12"], reader=lambda: ACCEPTANCE_RUN_AVAILABLE_BYTES, out=out)
    assert code == 1
    text = out.getvalue()
    assert "available 8,366,292,992 bytes (7.79 GB" in text
    assert "need 12.00 GB (12,884,901,888 bytes)" in text
    assert text.strip().endswith("refused (exit 1)")


def test_main_exit_2_when_the_reader_returns_no_number():
    out = io.StringIO()
    assert guard.main(["--need-gb", "12"], reader=lambda: None, out=out) == 2
    assert out.getvalue().startswith("free memory guard: ")


def test_main_exit_2_on_a_malformed_or_missing_argument(capsys):
    out = io.StringIO()
    assert guard.main(["--need-gb", "twelve"], reader=lambda: 20 * GB, out=out) == 2
    assert guard.main([], reader=lambda: 20 * GB, out=out) == 2
    assert guard.main(["--need-gb", "-3"], reader=lambda: 20 * GB, out=out) == 2
    assert "at or above 0" in out.getvalue()
    capsys.readouterr()  # argparse's usage text on stderr, not a failure


def test_reader_is_not_called_when_parsing_fails():
    calls = []

    def reader():
        calls.append(1)
        return 20 * GB

    assert guard.main(["--need-gb"], reader=reader, out=io.StringIO()) == 2
    assert calls == []


def test_psutil_reader_returns_a_byte_count():
    n = guard.psutil_available_bytes()
    assert isinstance(n, int)
    assert n >= 0


def test_command_line_exit_code_matches_verdict():
    """The module runs as ``python -m`` from Source/Python; a 0 GB line holds on any machine (exit 0)."""
    proc = subprocess.run([sys.executable, "-m", "mimo_bridge.tools.free_memory_guard", "--need-gb", "0"],
                          capture_output=True, text=True, cwd=SOURCE_PYTHON_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("free memory guard: available ")
    assert proc.stdout.strip().endswith("allowed (exit 0)")


def test_command_line_exit_2_on_a_malformed_argument():
    """A malformed ``--need-gb`` exits 2 as a command too, so an ``&&`` chain stops on it."""
    proc = subprocess.run([sys.executable, "-m", "mimo_bridge.tools.free_memory_guard", "--need-gb", "x"],
                          capture_output=True, text=True, cwd=SOURCE_PYTHON_ROOT)
    assert proc.returncode == 2, proc.stdout
