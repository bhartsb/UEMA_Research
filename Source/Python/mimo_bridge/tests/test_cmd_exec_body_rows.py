# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The trainer's executed-order size gauges (stats field cmd_exec) average the BODY parts' rows and
leave the eyes' row out by its NAME, in both body layouts.

cmd_exec (added 2026-09-14, commit ab6caa254) reads each latched order: its mean strength (0 = limp,
1 = full) and the mean distance in cm between each part's ordered position and where the part is
now. Until 2026-09-23 it averaged every row, the eyes' row included. The decoder gives the eyes' row
an empty joint chain (action_decoder.py), so that row's strength and position reach no actuator;
the eyes are driven by the gaze output instead. The eyes' row is row 23 in both layouts, and the
last row is the chest in trunk_v1 (every H84 segment), so "leave out the last row" would drop the
chest there. The first four tests build no world; the last drives the real trainer on CPU with one
body and no vision, in both layouts.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for

LAYOUTS = ["legacy", "trunk_v1"]
HANDS = {"wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
         "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R"}


def _order_and_pose(names, seed=0):
    rng = np.random.default_rng(seed)
    held = rng.uniform(0.0, 1.0, (3, len(names), 9))
    held[..., 0:3] = rng.uniform(-40.0, 40.0, (3, len(names), 3))       # ordered positions, cm
    achieved = rng.uniform(-0.2, 0.2, (3, len(names), 10))             # in units of 200 cm
    gi = names.index("gaze")
    held[:, gi, 8] = 1.0                                               # full strength, far away:
    held[:, gi, 0:3] = achieved[:, gi, 0:3] * 200.0 + 150.0            # unlike every body row
    return held, achieved


def _step_cm(held, achieved, rows):
    return float(np.linalg.norm(held[:, rows, 0:3] - achieved[:, rows, 0:3] * 200.0, axis=-1).mean())


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_eyes_row_is_left_out_and_every_body_row_counts(layout):
    from mimo_bridge.train_mimo import cmd_exec_sample
    names = list(names_for(layout))
    gi = names.index("gaze")
    body = [i for i, n in enumerate(names) if n != "gaze"]
    assert len(body) == len(names) - 1
    held, achieved = _order_and_pose(names)
    out = cmd_exec_sample(held, achieved, names)
    assert out["strength"] == pytest.approx(held[:, body, 8].mean(), abs=1e-12)
    assert out["step_cm"] == pytest.approx(_step_cm(held, achieved, body), abs=1e-9)
    # the eyes' row changes nothing
    other = held.copy()
    other[:, gi, 8] = 0.0
    other[:, gi, 0:3] -= 300.0
    again = cmd_exec_sample(other, achieved, names)
    assert again["strength"] == pytest.approx(out["strength"], abs=1e-12)
    assert again["step_cm"] == pytest.approx(out["step_cm"], abs=1e-9)
    # every body row counts, the chest and lower trunk included: 0.3 more strength on one row
    # raises the mean by 0.3 / the number of body rows
    for i in body:
        up = held.copy()
        up[:, i, 8] += 0.3
        assert (cmd_exec_sample(up, achieved, names)["strength"] - out["strength"]
                == pytest.approx(0.3 / len(body), abs=1e-12)), names[i]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_hand_readings_are_the_wrists_and_fingers(layout):
    from mimo_bridge.train_mimo import cmd_exec_sample
    names = list(names_for(layout))
    assert set(names[:12]) == HANDS
    held, achieved = _order_and_pose(names, seed=1)
    out = cmd_exec_sample(held, achieved, names)
    assert out["strength_hands"] == pytest.approx(held[:, :12, 8].mean(), abs=1e-12)
    assert out["step_cm_hands"] == pytest.approx(_step_cm(held, achieved, list(range(12))), abs=1e-9)


def test_no_pose_gives_no_position_step():
    from mimo_bridge.train_mimo import cmd_exec_sample
    names = list(names_for("trunk_v1"))
    held, _ = _order_and_pose(names)
    out = cmd_exec_sample(held, None, names)
    assert out["step_cm"] is None and out["step_cm_hands"] is None
    assert out["strength"] == pytest.approx(held[:, [i for i, n in enumerate(names) if n != "gaze"], 8].mean())


def test_segment_31_stored_rows_read_the_body_row_numbers():
    """The 16 (moment, body) rows segment 31 lived at ckpt_step_344400, stored with their
    layout: 0.4876 strength and a 30.234 cm step over the body rows; every row including the
    eyes' row reads 0.4864 and 30.132 cm."""
    from mimo_bridge.train_mimo import cmd_exec_sample
    path = (Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v3" / "measurements"
            / "motor_rows_seg31_ckpt_step_344400.npz")
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    assert names == list(names_for("trunk_v1"))
    order, ach = d["order"].astype(np.float64), d["ach"].astype(np.float64)
    out = cmd_exec_sample(order, ach, names)
    assert round(out["strength"], 4) == 0.4876
    assert round(out["step_cm"], 3) == 30.234
    assert round(float(order[..., 8].mean()), 4) == 0.4864      # the reading before the fix
    assert round(_step_cm(order, ach, list(range(len(names)))), 3) == 30.132


def _flags(tmp_path, layout):
    return ['--out', str(tmp_path / 'run'), '--device', 'cpu', '--n-envs', '1', '--no-vision',
            '--curriculum', 'off', '--age-binding', 'none', '--delay-binding', 'off', '--episode-steps', '0',
            '--action-repeat', '1', '--motor-hold', '1', '--minutes', '0', '--max-steps', '4', '--no-babble',
            '--self-state', 'perceive', '--body-function', 'modulate', '--persistent-memory', 'on',
            '--imagination', 'off', '--next-frame-loss', 'on', '--contact-profile', 'mechanical_v4',
            '--replay', 'off', '--report-every', '1', '--video-every-min', '0',
            '--body-layout', layout, '--decoder-kp', '6.0',
            '--checkpoint-every-min', '0', '--checkpoint-storage', str(tmp_path / 'storage' / 'checkpoints')]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_trainer_records_the_body_rows_of_each_latched_order(tmp_path, monkeypatch, layout):
    """Driven through the real trainer on CPU: one body, one order latched per step (--motor-hold
    1), one stats record per step (--report-every 1). The eyes' row of every order is set to full
    strength right after the model makes it; the decoder never reads that row's strength, and a
    gauge that counted the row would read above the body rows' mean."""
    import torch
    from mimo_bridge import lem_model
    from mimo_bridge.tools import preflight
    from mimo_bridge.train_mimo import main
    monkeypatch.setattr(preflight, 'run_probes', lambda path, env_kw=None: dict(ok=True, checks=[]))
    torch.set_num_threads(1)
    real = lem_model.LemLite.step
    seen = []

    def spy(self, *a, **k):
        res = real(self, *a, **k)
        names = list(self.handle_names)
        res.motor_for_action[:, names.index('gaze'), 8] = 1.0
        seen.append((np.array(res.motor_for_action, dtype=np.float64), names))
        return res

    monkeypatch.setattr(lem_model.LemLite, 'step', spy)
    assert main(_flags(tmp_path, layout)) == 0
    records = [json.loads(line) for line in
               (tmp_path / 'run' / 'replay_training_stats.jsonl').read_text().splitlines()]
    got = [r['cmd_exec']['strength_mean'] for r in records if r['type'] == 'mimo' and r.get('cmd_exec')]
    assert seen and len(got) == len(seen)
    for (held, names), reading in zip(seen, got):
        assert names == list(names_for(layout)) and held.shape[1] == len(names)
        body = [i for i, n in enumerate(names) if n != 'gaze']
        assert reading == round(float(held[:, body, 8].mean()), 4)
        assert reading != round(float(held[..., 8].mean()), 4)      # the all-row reading
