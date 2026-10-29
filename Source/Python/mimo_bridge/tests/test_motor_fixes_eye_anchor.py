# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The anchored eye reading's program for the eye fixes' deploy (docs/gain_of_function_v4/motor_fixes_eye_anchor.py; the
fourth review's finding 4, fixed in its fix round). Its measurement is the ability scheduler's own gaze probe
(probe_saved_gaze.py, run on a checkpoint in a pause job); pinned here without a checkpoint:
  * the switched reading rewrites every restored body's stored eye rows right after its episode is restored, with the
    trainer's own switch, and refuses a load that recorded no convention change (a checkpoint already trained under
    the convention asked for would otherwise be read as switched);
  * every reading gets a folder of its own under --out-root, named by run, checkpoint and convention, so a pause job
    can run again and the watcher's --anchor-readings finds it.
"""
import importlib.util
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4"
_spec = importlib.util.spec_from_file_location("motor_fixes_eye_anchor", DOCS / "motor_fixes_eye_anchor.py")
EA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EA)


class _Base:
    def __init__(self, model, body):
        self.model, self.executed, self.episode_body = model, "ledger of body %d" % body, dict(body=body)


def test_the_switched_reading_rewrites_every_restored_bodys_rows_and_refuses_no_change():
    calls = []

    def switch(model, executed):
        calls.append((model, executed))
        return executed + " (rewritten)", {"replay.executed": 8, "trainer.prev_executed": 1}
    S = EA.switched_session(_Base, switch)
    a, b = S("model", 0), S("model", 3)
    assert a.executed == "ledger of body 0 (rewritten)" and b.executed == "ledger of body 3 (rewritten)"
    assert calls == [("model", "ledger of body 0"), ("model", "ledger of body 3")]
    assert [r["body"] for r in S.rewrites] == [0, 3] and S.rewrites[0]["rewritten"]["replay.executed"] == 8
    none = EA.switched_session(_Base, lambda model, executed: (executed, None))
    with pytest.raises(RuntimeError, match="no eye row convention change"):
        none("model", 1)


def test_each_reading_gets_a_folder_of_its_own(tmp_path):
    ck = "C:/runs/H84_fresh_drivers_20260918_seg40_resume_x/checkpoints/ckpt_step_415075"
    got = EA.out_folder(tmp_path, ck, "physical")
    assert got.parent == tmp_path
    assert re.fullmatch(r"H84_fresh_drivers_20260918_seg40_resume_x__ckpt_step_415075__physical__\d{8}_\d{6}", got.name)
    assert EA.out_folder(tmp_path, ck, None).name.split("__")[2] == "own"
