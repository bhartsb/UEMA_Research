# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The native action writer limits only the UE action frame's 23 limb rows.

lem/native_ops.py carries the C++ inference loop as a string. Its action writer thread
applies the developmental limits (position times rom_limit x position_range_cm, radius
and strength clamps) to each motor row before writing the frame to UE. The frame holds
23 limb rows of 9 floats, then a gaze field (2 floats) and a voice field (80 floats).

From 2026-08-24 (the eyes became handle 23, body.n_handles 24) to 2026-09-23 that loop
ran over n_handles rows, so its 24th pass took the gaze field and the first 7 voice
numbers for a limb row: gaze angles and voice[0] were multiplied by up to 200 and sent to
UE, and the motor stats' largest strength read voice[6].

These tests compile the C++ block that now holds the rule (BEGIN/END action_frame_rows in
native_ops.CPP_SOURCE) on its own with MSVC - the full extension needs CUDA and a world -
and run frames laid out as the writer lays them out, for the UE config's 24-handle body,
the older 23-handle body and the MIMo trunk_v1 body (26 rows).
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from lem import native_ops
from lem.action import writer as AW
from lem.model.body_sets import HANDLE_NAMES
from mimo_bridge.body_layout import names_for

FRAME_FLOATS = 1280 // 4          # pinned_action_bufs[slot] holds 1280 bytes
GAZE_AT = 207                     # the writer's memcpy offsets
VOICE_AT = 209


def _cpp_block(source: str | None = None) -> str:
    src = native_ops.CPP_SOURCE if source is None else source
    m = re.search(r"// BEGIN action_frame_rows.*?\n(.*?)// END action_frame_rows", src, re.S)
    assert m, "the action_frame_rows block is not in native_ops.CPP_SOURCE"
    return m.group(1)


def _compile(block: str, out_dir) -> ctypes.CDLL:
    """Compile the block into a DLL with MSVC; skip where no MSVC is installed."""
    if sys.platform != "win32" or not native_ops._setup_msvc_env():
        pytest.skip("MSVC not available to compile the C++ block")
    cpp = os.path.join(str(out_dir), "action_frame_rows.cpp")
    dll = os.path.join(str(out_dir), "action_frame_rows.dll")
    with open(cpp, "w", encoding="utf-8") as f:
        f.write("#include <algorithm>\n#include <cstdint>\n" + block + r"""
extern "C" __declspec(dllexport) float frame_limits(float* frame, int64_t n_handles,
        float rom, float strength, float position_range_cm) {
    return apply_action_frame_limits(frame, n_handles, rom, strength, position_range_cm);
}
extern "C" __declspec(dllexport) int frame_rows(int64_t n_handles) {
    return action_frame_limb_rows(n_handles);
}
""")
    r = subprocess.run(["cl", "/nologo", "/O2", "/EHsc", "/LD", cpp, f"/Fe{dll}",
                        f"/Fo{os.path.join(str(out_dir), 'action_frame_rows.obj')}"],
                       capture_output=True, text=True, cwd=str(out_dir))
    assert r.returncode == 0, r.stdout + r.stderr
    lib = ctypes.CDLL(dll)
    lib.frame_limits.restype = ctypes.c_float
    lib.frame_limits.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
                                 ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.frame_rows.restype = ctypes.c_int
    lib.frame_rows.argtypes = [ctypes.c_int64]
    return lib


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return _compile(_cpp_block(), tmp_path_factory.mktemp("action_frame_rows"))


def _frame(n_handles: int, seed: int = 0):
    """A frame as the writer thread builds it: min(n_handles*9, 207) motor floats, the
    gaze at +207, the voice at +209. Motor values run past every clamp."""
    g = np.random.default_rng(seed)
    buf = np.zeros(FRAME_FLOATS, np.float32)
    motor = g.uniform(-2.0, 2.0, (n_handles, 9)).astype(np.float32)
    n_mot = min(n_handles * 9, 207)
    buf[:n_mot] = motor.reshape(-1)[:n_mot]
    buf[GAZE_AT:GAZE_AT + 2] = [0.2, -0.1]                       # radians
    buf[VOICE_AT:VOICE_AT + 80] = np.linspace(0.05, 1.5, 80, dtype=np.float32)
    return buf


def _run(lib, buf, n_handles, rom=0.6, strength=0.8, prc=200.0):
    out = buf.copy()
    max_str = lib.frame_limits(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               n_handles, rom, strength, prc)
    return out, max_str


def _expected_limb_rows(buf, n_rows, rom=0.6, strength=0.8, prc=200.0):
    rows = buf[:n_rows * 9].reshape(n_rows, 9).copy()
    rows[:, 0:3] = np.clip(rows[:, 0:3], -1, 1) * rom * prc
    rows[:, 7] = np.clip(rows[:, 7], 0, 1)
    rows[:, 8] = np.clip(rows[:, 8], 0, strength)
    return rows


def _ue_config_n_handles() -> int:
    from pathlib import Path
    import lem
    from lem.config.loader import load_and_resolve
    cfg = load_and_resolve(str(Path(lem.__file__).parent / "master_config.yaml"),
                           skip_validation=True)
    return int(cfg["body"]["n_handles"])


# The bodies this writer can be handed: the UE config (lem/master_config.yaml
# body.n_handles, 24 = the legacy layout with the eyes), the body before the eyes were
# added, and the MIMo trunk_v1 body.
BODIES = {
    "ue_config": _ue_config_n_handles(),
    "legacy_24": len(names_for("legacy")),
    "before_eyes_23": HANDLE_NAMES.index("gaze"),
    "trunk_v1_26": len(names_for("trunk_v1")),
}


@pytest.mark.parametrize("body", sorted(BODIES))
def test_the_gaze_and_voice_fields_reach_ue_as_the_model_produced_them(lib, body):
    n = BODIES[body]
    buf = _frame(n)
    out, _ = _run(lib, buf, n)
    np.testing.assert_array_equal(out[GAZE_AT:GAZE_AT + 2], buf[GAZE_AT:GAZE_AT + 2],
                                  err_msg="the gaze angles were rescaled as a limb row")
    np.testing.assert_array_equal(out[VOICE_AT:], buf[VOICE_AT:],
                                  err_msg="the voice numbers were rescaled as a limb row")


@pytest.mark.parametrize("body", sorted(BODIES))
def test_every_limb_row_gets_the_developmental_limits(lib, body):
    """The rows limited are the limbs, wrist_L to foot_R - every name before 'gaze'."""
    n = BODIES[body]
    limbs = HANDLE_NAMES.index("gaze")
    buf = _frame(n, seed=1)
    out, max_str = _run(lib, buf, n)
    want = _expected_limb_rows(buf, limbs)
    np.testing.assert_allclose(out[:limbs * 9].reshape(limbs, 9), want, rtol=1e-6)
    assert max_str == pytest.approx(float(want[:, 8].max()))


@pytest.mark.parametrize("body", sorted(BODIES))
def test_the_stats_largest_strength_comes_from_a_limb_row(lib, body):
    """voice[6] sits where a 24th row's strength would; it must not be read as one."""
    n = BODIES[body]
    limbs = HANDLE_NAMES.index("gaze")
    buf = _frame(n, seed=3)
    buf[8:limbs * 9:9] = np.linspace(0.1, 0.5, limbs, dtype=np.float32)
    buf[VOICE_AT + 6] = 0.7                                     # under the 0.8 limit
    _, max_str = _run(lib, buf, n)
    assert max_str == pytest.approx(0.5), (
        f"largest strength {max_str:.2f}, the largest on a limb row is 0.50")


def test_a_body_with_fewer_rows_than_the_frame_limits_only_its_own_rows(lib):
    buf = _frame(20, seed=2)
    out, _ = _run(lib, buf, 20)
    np.testing.assert_allclose(out[:180].reshape(20, 9), _expected_limb_rows(buf, 20),
                               rtol=1e-6)
    np.testing.assert_array_equal(out[180:], buf[180:])


def test_the_row_bound_is_the_frames_not_the_models(lib):
    assert [lib.frame_rows(n) for n in (0, 20, 23, 24, 26)] == [0, 20, 23, 23, 23]


# ------------------------------------------------------------------ the wiring
def test_the_writer_thread_uses_the_rule_and_no_loop_of_its_own():
    src = native_ops.CPP_SOURCE
    writer = src[src.index("auto writer_fn = [&]() {"):src.index("std::thread action_writer_thread")]
    assert "apply_action_frame_limits(" in writer
    assert not re.search(r"for\s*\(\s*int\s+h\s*=\s*0\s*;\s*h\s*<\s*n_handles", writer), (
        "a loop bounded by the model's n_handles is back in the action writer")


def test_the_frame_row_count_agrees_everywhere():
    m = re.search(r"constexpr int ACTION_FRAME_LIMB_ROWS\s*=\s*(\d+);", native_ops.CPP_SOURCE)
    assert m
    assert int(m.group(1)) == native_ops.ACTION_FRAME_LIMB_ROWS == 23
    assert (AW.OFF_GAZE - AW.OFF_MOTOR) // (9 * 4) == native_ops.ACTION_FRAME_LIMB_ROWS
    assert HANDLE_NAMES[native_ops.ACTION_FRAME_LIMB_ROWS] == "gaze", (
        "the frame's limb rows are the body's rows before the eyes")


# ------------------------------------------------------------ the start-up check
class _Bundle:
    def __init__(self, n_handles):
        self.n_handles = n_handles


def test_a_body_too_small_for_the_frame_is_refused_before_the_loop(monkeypatch):
    def _loader():
        raise AssertionError("reached the C++ loader with a body smaller than the frame")
    monkeypatch.setattr(native_ops, "load_native_ops", _loader)
    with pytest.raises(ValueError, match="at least 23 body handles"):
        native_ops.run_native_inference_loop(
            _Bundle(20), None, 0, 0, None, None, None, (None, None), (None, None),
            "", "", None, {}, None)


@pytest.mark.parametrize("body", sorted(BODIES))
def test_bodies_that_fill_the_frame_pass_the_check(monkeypatch, body):
    monkeypatch.setattr(native_ops, "load_native_ops", lambda: None)
    with pytest.raises(RuntimeError, match="extension not available"):
        native_ops.run_native_inference_loop(
            _Bundle(BODIES[body]), None, 0, 0, None, None, None, (None, None),
            (None, None), "", "", None, {}, None)
