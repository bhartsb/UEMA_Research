# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Beside a running trainer the scheduler keeps room for the trainer's checkpoint save (Oct 1 2026, 10:52 AM: a
12 GiB test fitted under the plain 6 GiB reserve, and with an hourly save left 0.8 GiB of commit room)."""
from mimo_bridge.tools import ability_rechecks as AR
from mimo_bridge.tools.commit_guard import verdict


def test_a_12_gib_test_does_not_fit_beside_training_at_the_morning_commit():
    committed, limit = 65.3, 84.4                      # 9:25 AM reading with the trainer and 8 workers up
    assert not verdict(committed, limit, 12, AR.beside_training_reserve(False))
    assert verdict(committed, limit, 12, AR.beside_training_reserve(True))   # a saved window has the room


def test_the_reserve_beside_training_includes_the_save_peak():
    assert AR.beside_training_reserve(False) == 6.0 + AR.SAVE_PEAK_GIB
    assert AR.beside_training_reserve(True) == 6.0
