# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Refuse a command when the machine's available physical memory is under a stated line.

Input: ``--need-gb N``, the memory line in GB, where 1 GB = 1,073,741,824 bytes (2 to the power 30, the
GB that Windows Task Manager and macOS Activity Monitor show). The reading is
``psutil.virtual_memory().available``, an int of bytes (psutil 7.2.2, requirements.txt:13): on Windows the
available physical memory that ``GlobalMemoryStatusEx`` reports (psutil/_pswindows.py:188-190, ``avail =
availphys``); on macOS the free pages plus the inactive pages (psutil/_psosx.py:97, ``avail = inactive +
free``). Either number is the physical memory a new process can take before the operating system pages.

Operation: ``available_bytes >= need_gb * 1,073,741,824`` is the only comparison.

Output: one line on standard output naming the available bytes, the line and the verdict, then exit
code 0 when the comparison holds and 1 when it does not (2 for a malformed ``--need-gb`` or a reader that
returns no byte count). Joined with ``&&`` before a pytest run, a refused guard runs nothing
(UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04, section 4: "Every pytest run follows the memory guard of
WI-1 joined with &&"):

    python -m mimo_bridge.tools.free_memory_guard --need-gb 12 && python -m pytest ... --junitxml=<file>

That chain is the Git Bash and zsh form. Windows PowerShell 5.1 has no ``&&``; there the second command
runs under ``if ($LASTEXITCODE -eq 0) { ... }``.

The reader of the memory number is injected (``main(argv, reader=...)``, ``verdict(need_gb,
available_bytes)``) so a test checks both exit codes without changing the machine's memory.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TextIO

BYTES_PER_GB = 1024 ** 3  # 1,073,741,824 bytes


def psutil_available_bytes() -> int:
    """Return ``psutil.virtual_memory().available``, the machine's available physical memory in bytes.

    Input: none (the kernel's memory status at the moment of the call). Output: an int of bytes, at
    least 0. Imported inside the function so the module's own import never needs psutil.
    """
    import psutil

    return int(psutil.virtual_memory().available)


@dataclass(frozen=True)
class Verdict:
    """The guard's result: the line asked for, the bytes read and whether the line holds.

    ``need_bytes`` = ``need_gb`` x 1,073,741,824; ``allowed`` = ``available_bytes >= need_bytes``.
    """

    need_gb: float
    need_bytes: int
    available_bytes: int
    allowed: bool

    @property
    def exit_code(self) -> int:
        """0 when the line holds, 1 when it does not (the ``&&`` chain stops on 1)."""
        return 0 if self.allowed else 1

    def line(self) -> str:
        """One line for standard output: both numbers with units, the verdict and the exit code."""
        state = "allowed (exit 0)" if self.allowed else "refused (exit 1)"
        return (
            "free memory guard: available %s bytes (%s GB of 1,073,741,824 bytes); need %s GB "
            "(%s bytes); %s"
            % (f"{self.available_bytes:,}", _gb_text(self.available_bytes), _gb_text(self.need_bytes),
               f"{self.need_bytes:,}", state)
        )


def _gb_text(n_bytes: int) -> str:
    """Bytes as GB with two decimals, trailing zeros kept ("12.00"), for the output line only."""
    return "%.2f" % (n_bytes / BYTES_PER_GB)


def verdict(need_gb: float, available_bytes: int) -> Verdict:
    """Compare ``available_bytes`` against ``need_gb`` GB.

    Input: ``need_gb`` at or above 0 (a float; 12 means 12,884,901,888 bytes); ``available_bytes`` an int at
    or above 0. Operation: ``need_bytes = int(need_gb * 1,073,741,824)``; ``allowed = available_bytes >=
    need_bytes``. Output: a ``Verdict``. Raises ``ValueError`` for a negative or non-finite ``need_gb`` or a
    negative ``available_bytes``.
    """
    need_gb = float(need_gb)
    if not math.isfinite(need_gb) or need_gb < 0:
        raise ValueError("--need-gb must be a finite number at or above 0, not %r" % (need_gb,))
    if available_bytes < 0:
        raise ValueError("available_bytes must be at or above 0, not %r" % (available_bytes,))
    need_bytes = int(need_gb * BYTES_PER_GB)
    return Verdict(need_gb=need_gb, need_bytes=need_bytes, available_bytes=int(available_bytes),
                   allowed=int(available_bytes) >= need_bytes)


def build_parser() -> argparse.ArgumentParser:
    """The command line: ``--need-gb`` (required, GB as a float)."""
    parser = argparse.ArgumentParser(
        prog="python -m mimo_bridge.tools.free_memory_guard",
        description="Exit 0 when psutil.virtual_memory().available is at or above --need-gb GB "
                    "(1 GB = 1,073,741,824 bytes), else exit 1.")
    parser.add_argument("--need-gb", type=float, required=True,
                        help="the available physical memory the following command needs, in GB")
    return parser


def main(argv: Optional[Sequence[str]] = None,
         reader: Callable[[], int] = psutil_available_bytes,
         out: Optional[TextIO] = None) -> int:
    """Run the guard.

    Input: ``argv`` (default ``sys.argv[1:]``), ``reader`` (returns available bytes), ``out`` (a text
    stream, default ``sys.stdout``). Operation: parse, read, ``verdict``. Output: the verdict's line on
    ``out`` and its exit code (0 allowed, 1 refused, 2 for a malformed argument or a reader whose return
    value is not a number, which ``verdict`` raises as ``TypeError``).
    """
    out = sys.stdout if out is None else out
    parser = build_parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as exc:  # argparse already printed the usage line
        return 2 if exc.code else 0
    try:
        result = verdict(args.need_gb, reader())
    except (ValueError, TypeError) as exc:
        print("free memory guard: %s" % exc, file=out)
        return 2
    print(result.line(), file=out)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
