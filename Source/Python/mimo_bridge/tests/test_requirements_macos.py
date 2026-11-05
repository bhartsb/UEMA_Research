# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""requirements-macos.txt carries the same pins as requirements.txt, with torch pinned from the default index.

Inputs: the two requirements files at the checkout root (Source/Python/mimo_bridge/tests -> parents[4]).
Operation: parse every non-comment line as ``name==version``; compare the two name sets and the shared versions.
Outputs: pytest pass or fail lines; the download test runs ``pip download`` for the macOS arm64 platform only when
UEMA_NETWORK_TESTS=1 (it needs the network) and counts the wheels it fetched. The platform tag is macosx_13_0_arm64,
the set's install floor (opencv-python-headless 5.0.0.93 ships only a macosx_13_0_arm64 wheel; pip expands the tag to
every older macOS minor, so the 11_0 and 12_0 wheels resolve too); with macosx_11_0_arm64 pip stops at scipy 1.18.0.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
WINDOWS_FILE = REPO / "requirements.txt"
MACOS_FILE = REPO / "requirements-macos.txt"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.]+)$")


def pins(path: Path) -> dict[str, str]:
    """Return {package name: version} from one requirements file.

    Inputs: the file path. Operation: each line stripped; blank lines and lines starting with ``#`` dropped; the rest
    must match ``name==version`` (any other line, an index URL for example, fails the test). Output: the dict, in
    file order; a name repeated in the file fails the test.
    """
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN.match(line)
        assert m, f"{path.name}: line is not name==version: {line!r}"
        name = m.group(1).lower().replace("_", "-")
        assert name not in out, f"{path.name}: {name} pinned twice"
        out[name] = m.group(2)
    return out


def test_macos_pins_are_the_windows_pins_plus_torch():
    """The name set equals requirements.txt's pins plus torch: requirements.txt installs torch by its header line
    (``pip install torch==2.6.0 --index-url .../cu124``), not by a pin, and the macOS file pins it instead."""
    windows = pins(WINDOWS_FILE)
    macos = pins(MACOS_FILE)
    assert "torch" not in windows, "requirements.txt now pins torch; drop the +torch allowance here"
    assert set(macos) == set(windows) | {"torch"}, (
        f"missing on macOS: {sorted(set(windows) - set(macos))}; extra on macOS: "
        f"{sorted(set(macos) - set(windows) - {'torch'})}"
    )
    assert len(macos) == 22
    for name, version in windows.items():
        assert macos[name] == version, f"{name}: macOS pins {macos[name]}, requirements.txt pins {version}"


def test_torch_pin_matches_the_cuda_header_line():
    """The torch version comes from requirements.txt's own header line, so one file cannot drift from the other."""
    header = WINDOWS_FILE.read_text(encoding="utf-8")
    m = re.search(r"pip install torch==([0-9.]+) --index-url", header)
    assert m, "requirements.txt header no longer names the torch version"
    assert pins(MACOS_FILE)["torch"] == m.group(1)


def test_macos_header_states_the_install_facts():
    """The header names the native arm64 Python, the two compiled wheels, the MIMo install line, mjpython and the two
    published MPS floors with the 2.6.0 floor marked unverified. The comment lines are joined with one space before the
    search (a ``#`` continuation and its indent become one space), so re-wrapping the header does not fail the test."""
    text = MACOS_FILE.read_text(encoding="utf-8")
    header = " ".join(line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#"))
    for needle in (
        "arm64",
        "mujoco-3.3.0-cp313-cp313-macosx_11_0_arm64.whl",
        "torch-2.6.0-cp313-none-macosx_11_0_arm64.whl",
        "pip install -e third_party/MIMo/mimo --no-deps",
        "mjpython",
        "the current macOS version is not 15.0+",
        "MacOS-14 or later",
        "unverified",
        "macosx_13_0_arm64",
    ):
        assert needle in header, f"requirements-macos.txt header lacks {needle!r}"
    assert "--index-url" not in "\n".join(
        l for l in text.splitlines() if not l.lstrip().startswith("#")
    ), "requirements-macos.txt must not point pip at another index"


@pytest.mark.skipif(
    os.environ.get("UEMA_NETWORK_TESTS") != "1",
    reason="needs the network: set UEMA_NETWORK_TESTS=1 to run pip download for macosx_13_0_arm64",
)
def test_every_pin_has_a_macos_arm64_wheel(tmp_path: Path):
    """pip download with the macOS 13.0 arm64 platform and Python 3.13, binaries only, no dependencies: exit 0 and 22
    wheels in the folder (one per pin), the acceptance line of the plan's WI-10 row with the platform tag raised from
    11_0 to 13_0 (measured Nov 4, 2026: 11_0 fails at scipy==1.18.0, whose lowest arm64 wheel is macosx_12_0)."""
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--platform", "macosx_13_0_arm64", "--python-version", "3.13", "--only-binary=:all:", "--no-deps",
        "-r", str(MACOS_FILE), "-d", str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"pip download did not finish within {exc.timeout:.0f} s (the builder's run took 5.84 s): {cmd}")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    wheels = sorted(p.name for p in tmp_path.glob("*.whl"))
    assert len(wheels) == 22, wheels
    for name in pins(MACOS_FILE):
        stem = name.replace("-", "_")
        assert any(w.lower().startswith(stem + "-") for w in wheels), f"no wheel for {name}: {wheels}"
