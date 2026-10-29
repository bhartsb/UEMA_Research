# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""tools/command_following.py counts a trial only when the gauge scored a part in it.

What was wrong (found 2026-09-23): env._command_response returns cos 0.0 with n 0 when it scores
no part, and command_following read cos without looking at n, so such a trial went into every mean
as a direction match of 0. Under trunk_v1 (every H84 segment) the committed gauge took the LAST row,
the chest, for the eyes' row and never scored it; each chest trial orders only the chest, so every
chest trial read 0. The 2026-09-18 anchor in Saved/mimo_bridge/scale_anchors.json has chest = 0.0
in all 20 cells, 2 of each cell's 30 trials, and that table picked H84's decoder gain 6.

The tool builds trunk_v1 worlds only (it orders the chest and the lower trunk, rows the 24-row
legacy layout does not have), so there is no legacy case to test. These tests replace the world
with a stand-in whose gauge scores every ordered row except one skipped row, the way env.py does:
the committed gauge skipped the last row (the chest under trunk_v1), the fixed one skips the eyes'
row by name. No MuJoCo world is built.
"""
import json
import sys
import types

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import command_following as cf

COS = 0.8          # the direction match the stand-in gauge reports for any scored trial


def _stand_in_world(monkeypatch, skip):
    """Install a stand-in mimo_bridge.env whose BridgeEnv gauge skips the row `skip(names)`."""
    built = []

    class FakeEnv:
        def __init__(self, **kw):
            assert kw["body_layout"] == "trunk_v1"
            self.names = list(names_for(kw["body_layout"]))
            rows = len(self.names)
            self.proprio = types.SimpleNamespace(body_ids=list(range(rows)), handle_names=tuple(self.names))
            self.data = types.SimpleNamespace(xpos=np.zeros((rows, 3)),
                                              xquat=np.tile([1.0, 0.0, 0.0, 0.0], (rows, 1)))
            self.skip = skip(self.names)
            built.append(self)

        def reset(self):
            pass

        def close(self):
            pass

        def step(self, order, *_a, render=False):
            scored = [i for i in range(len(self.names)) if order[i, 8] > 0 and i != self.skip]
            n = len(scored)
            cr = {"cos": COS if n else 0.0, "ratio": 0.5 if n else 0.0, "n": n,
                  "pos_cos": 0.3 if n else 0.0}
            return types.SimpleNamespace(info={"cmd_response_slow": cr, "decoded_ctrl": np.zeros(8)})

    mod = types.ModuleType("mimo_bridge.env")
    mod.BridgeEnv = FakeEnv
    monkeypatch.setitem(sys.modules, "mimo_bridge.env", mod)
    return built


def _run(trials=30):
    return cf.run(6.0, 0.5, trials, 20.0, np.random.default_rng(0), "crib", 24.0, 25, quiet=True)


def test_a_trial_the_gauge_scored_no_part_for_is_left_out_not_counted_as_zero(monkeypatch):
    # the committed gauge under trunk_v1: the last row (the chest) taken for the eyes' row
    built = _stand_in_world(monkeypatch, skip=lambda names: len(names) - 1)
    out = _run()
    assert built[0].names[built[0].skip] == "chest", "the premise: the committed gauge skipped the chest"
    groups = len(cf.HANDLE_GROUPS)
    chest_trials = sum(1 for t in range(30) if list(cf.HANDLE_GROUPS)[t % groups] == "chest")
    assert chest_trials == 2
    assert out["per_group"]["chest"] is None, "no chest trial was scored, so the chest has no reading"
    assert out["trials_unscored"] == chest_trials
    assert out["per_group_unscored"]["chest"] == chest_trials
    assert all(c == 0 for g, c in out["per_group_unscored"].items() if g != "chest")
    assert out["trials"] == 30 - chest_trials
    # every scored trial read COS; before the fix the two chest zeros pulled the cell to 28/30 of it
    assert out["direction_match_mean"] == pytest.approx(COS)
    assert out["direction_match_p25"] == pytest.approx(COS)
    assert out["rotation_achieved_share_mean"] == pytest.approx(0.5)


def test_with_the_gauge_that_skips_the_eyes_by_name_every_trial_counts(monkeypatch):
    built = _stand_in_world(monkeypatch, skip=lambda names: names.index("gaze"))
    out = _run()
    assert built[0].skip == 23
    assert out["trials_unscored"] == 0 and out["trials"] == 30
    assert out["per_group"]["chest"] == pytest.approx(COS)
    assert out["direction_match_mean"] == pytest.approx(COS)


def test_a_cell_with_every_trial_unscored_reports_no_direction_match(monkeypatch):
    # a cell whose every trial orders the chest, on the gauge that never scored the chest
    monkeypatch.setattr(cf, "HANDLE_GROUPS", {"chest": ("chest",)})
    _stand_in_world(monkeypatch, skip=lambda names: len(names) - 1)
    out = cf.run(6.0, 0.5, 3, 20.0, np.random.default_rng(0), "crib", 24.0, 25, quiet=False)
    assert out["trials"] == 0 and out["trials_unscored"] == 3
    assert out["direction_match_mean"] is None and out["per_group"]["chest"] is None


def test_write_anchor_keeps_the_table_it_replaces_and_the_file_format(monkeypatch, tmp_path):
    _stand_in_world(monkeypatch, skip=lambda names: names.index("gaze"))
    old = {"measured": "2026-09-18T02:00:00-07:00", "best": {"kp": 6.0, "strength": 0.5},
           "defect_note": "chest cells read 0.0"}
    anchors = {"one_instant": 0.025, "command_following": old, "later": {"a": [1, 2]}}
    path = tmp_path / "Saved" / "mimo_bridge" / "scale_anchors.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cf.main(["--kps", "6", "--strengths", "0.5", "--trials", "15", "--write-anchor"])
    text = path.read_text(encoding="utf-8")
    new = json.loads(text)
    assert new["command_following"]["superseded"] == [old]
    assert new["command_following"]["table"][0]["trials_unscored"] == 0
    assert new["one_instant"] == 0.025 and new["later"] == {"a": [1, 2]}
    assert text == json.dumps(new, indent=2), "written in the file's own two-space format"
    # a second re-measurement keeps both earlier tables, newest first, one level deep
    cf.main(["--kps", "6", "--strengths", "0.5", "--trials", "15", "--write-anchor"])
    again = json.loads(path.read_text(encoding="utf-8"))["command_following"]
    assert [s["measured"] for s in again["superseded"]] == [new["command_following"]["measured"], old["measured"]]
    assert "superseded" not in again["superseded"][0]


# other anchors laid out by hand, the way following_out_of_100 is (inline lists, a CRLF line)
_HAND_LAID_BEFORE = '{\n  "one_instant": 0.025,\n  "mods": ["vision", "tactile"],\r\n  "command_following": '
_HAND_LAID_AFTER = (',\n  "following_out_of_100": {\n    "interval_95pct": [86.3, 88.5],'
                    ' "part_orders": 5000\n  }\n}')


def test_write_anchor_rewrites_only_its_own_entry_in_a_hand_laid_file(monkeypatch, tmp_path):
    """Until 2026-09-23 --write-anchor re-dumped the whole file, which re-laid every hand-laid anchor
    (on the day's file: 105,529 bytes written for 105,409 read, no value changed)."""
    _stand_in_world(monkeypatch, skip=lambda names: names.index("gaze"))
    old_entry = '{"measured": "2026-09-18T02:00:00-07:00", "best": {"kp": 6.0}}'
    path = tmp_path / "Saved" / "mimo_bridge" / "scale_anchors.json"
    path.parent.mkdir(parents=True)
    path.write_bytes((_HAND_LAID_BEFORE + old_entry + _HAND_LAID_AFTER).encode("utf-8"))
    monkeypatch.chdir(tmp_path)
    cf.main(["--kps", "6", "--strengths", "0.5", "--trials", "15", "--write-anchor"])
    text = path.read_bytes().decode("utf-8")
    assert text.startswith(_HAND_LAID_BEFORE), "every byte before the entry is kept"
    assert text.endswith(_HAND_LAID_AFTER), "every byte after the entry is kept"
    new = json.loads(text)
    assert new["command_following"]["superseded"] == [json.loads(old_entry)]
    assert new["following_out_of_100"] == {"interval_95pct": [86.3, 88.5], "part_orders": 5000}


@pytest.mark.parametrize("before", ['{\n  "one_instant": 0.025\n}', '{}'])
def test_write_anchor_adds_the_entry_when_the_file_has_none(monkeypatch, tmp_path, before):
    _stand_in_world(monkeypatch, skip=lambda names: names.index("gaze"))
    path = tmp_path / "Saved" / "mimo_bridge" / "scale_anchors.json"
    path.parent.mkdir(parents=True)
    path.write_text(before, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cf.main(["--kps", "6", "--strengths", "0.5", "--trials", "15", "--write-anchor"])
    new = json.loads(path.read_text(encoding="utf-8"))
    assert "superseded" not in new["command_following"]
    assert new["command_following"]["table"][0]["trials"] == 15
    assert {k: v for k, v in new.items() if k != "command_following"} == json.loads(before)
