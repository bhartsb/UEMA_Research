# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""follow_bout is the ONE-FRAME (10 ms) direction match, read on the last physics frame of each model
step in movement bouts (train_mimo.py follow_bout_ema from env cmd_response.cos), not the held order's
250 ms reading (cmd_response_cos_held). Until 2026-09-23 the old dashboard's chip showed it under the
250 ms help text ("over the whole life of each order", source cmd_response_cos_held) and the whole-life
page labelled it "over an order's life" (fix-rows-r2-dashboard.py).

The chip reads one scalar record field, so it has no body-layout case: trunk_v1 and the legacy layout
give the chip the same input."""
import html as _html
import importlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mimo_bridge.tools import dashboard
from mimo_bridge.tools import lineage_series as ls
from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, page_text


def _doc_keys(html_text):
    return re.findall(r'data-doc="([^"]+)"', html_text)


def test_the_bout_chip_wears_its_own_one_frame_help_and_the_held_chip_keeps_the_250_ms_help():
    chips = dashboard.chips_for({"follow_bout_win": 0.0792})
    keys = [_html.unescape(k) for k in _doc_keys(chips)]
    assert "chip:follow_bout" in keys and "chip:follow" not in keys, keys
    assert "10 ms" in page_text(chips)
    doc = dashboard.DOC["chip:follow_bout"]
    assert "10 ms" in doc["w"] and "last frame of each 50 ms model step" in doc["w"]
    assert "whole life" not in doc["w"] and "cmd_response_cos_held" not in doc["h"]
    assert "follow_bout" in doc["h"] and "stop rule" in doc["h"]
    # the held reading's chip (runs without movement bouts) keeps the 250 ms help
    held = dashboard.chips_for({"follow_slow": 0.14})
    held_keys = [_html.unescape(k) for k in _doc_keys(held)]
    assert "chip:follow" in held_keys and "chip:follow_bout" not in held_keys, held_keys
    assert "cmd_response_cos_held" in dashboard.DOC["chip:follow"]["h"]


def test_the_bout_chip_sets_the_reading_against_the_physics_line_on_its_own_frames(monkeypatch):
    # a stored one-frame reference: the chip names it and the distance, never the 250 ms goal line
    monkeypatch.setattr(ls, "FOLLOW", dict(ls.FOLLOW, frame_reachable=0.46, gauge_reachable=0.83))
    chips = page_text(dashboard.chips_for({"follow_bout_win": 0.0792}))
    assert "reads 46% on the same frames (38 percentage points short of it)" in chips, chips
    assert "83%" not in chips
    assert not bare_decimals(chips), bare_decimals(chips)
    # none stored: the chip says the goal line and the distance are unmeasured
    monkeypatch.setattr(ls, "FOLLOW", dict(ls.FOLLOW, frame_reachable=None, gauge_reachable=0.83))
    chips = page_text(dashboard.chips_for({"follow_bout_win": 0.0792}))
    assert "the goal line on these frames is unmeasured, so the distance to it is unmeasured" in chips, chips
    assert "83%" not in chips


def test_the_whole_life_page_names_the_one_frame_horizon_of_its_following_line(tmp_path, monkeypatch):
    from mimo_bridge.tools import lifetime
    assert "over an order's life" not in lifetime.YARDSTICKS["follow"][3]
    assert "10 ms" in lifetime.YARDSTICKS["follow"][3]
    GF = importlib.import_module("test_dashboard_gain_of_function")
    saved = tmp_path / "Saved" / "mimo_bridge"
    run = GF._run(saved, "P", [GF._rec(100 + i, timestamp=1_790_000_000 + i, follow_bout=0.08,
                                       cmd_response_cos_held=0.14, task_hand_fov=0.02, rest_over_bout_speed=0.88)
                               for i in range(10)])
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(run), "timestamp": 1}),
                                                      encoding="utf-8")
    for name, value in (("SAVED", str(saved)), ("EXP", str(saved / "experiments")),
                        ("OUT_HTML", str(saved / "lifetime.html")), ("OUT_JSON", str(saved / "lifetime.json"))):
        monkeypatch.setattr(lifetime, name, value)
    lifetime.build(max_age_s=10 ** 9, force=True)
    text = page_text((saved / "lifetime.html").read_text(encoding="utf-8"))
    card = text[text.index("Following a command"):]
    assert "8% of the way from unrelated to exact" in card, card[:400]
    assert "one 10 ms physics frame at the end of each 50 ms model step" in card
    assert "over an order's life" not in text
