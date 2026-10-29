# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The physical-episode companion survives a path past Windows' 260-character line (2026-09-18).

The companion file's path repeats the run name - once in the run directory, once in the
caregiver storage under it - and the H84 launch smokes' end save failed with "file not
found" on a folder that existed: Windows' refusal of a 280-character path. The writer
and reader now carry the extended-length prefix on that platform for any path of 240
characters or more; short paths and other platforms are untouched.
"""
import os
from pathlib import Path

from mimo_bridge import continuation_state as CS


def test_short_paths_pass_unchanged_and_long_windows_paths_get_the_prefix(tmp_path):
    short = tmp_path / "a.txt"
    assert CS.long_path(short) == os.fspath(short)
    long = tmp_path / ("x" * 250) / "a.txt"
    got = CS.long_path(long)
    if os.name == "nt":
        assert got.startswith("\\\\?\\") and got.endswith("a.txt")
        assert CS.long_path(got) == got, "the prefix is never doubled"
    else:
        assert got == os.fspath(long)


def test_a_companion_is_written_and_read_back_through_a_280_character_path(tmp_path):
    checkpoint = tmp_path / "ckpt_step_7"
    checkpoint.mkdir()
    deep = tmp_path
    while len(str(deep)) < 270:
        deep = deep / ("run_name_repeated_" + "z" * 40)
    state = dict(version=CS.VERSION, worlds=[dict(payload=dict(sim_time=1.5))], learner={},
                 trainer=dict(step=7), config={})
    receipt = CS.write_episode(checkpoint, state, storage_root=str(deep), published_path=None)
    assert receipt["step"] == 7 and len(receipt["storage_path"]) > 260
    back = CS.load(checkpoint)
    assert back["trainer"]["step"] == 7 and back["worlds"][0]["payload"]["sim_time"] == 1.5
