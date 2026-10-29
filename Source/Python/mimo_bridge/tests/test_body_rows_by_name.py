# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Code that picks a body part finds it by NAME, in both body layouts.

The trunk rows (lower trunk, chest) come AFTER the eyes' row (row 23): the 24-row legacy layout ends
with the eyes, the 26-row trunk_v1 layout (every run from H71 on) ends with the chest. A fixed row
number or "the first 24 rows" therefore means different body parts in the two layouts. Found
2026-09-23 beside the following-gauge defect (docs/gain_of_function_v3/agents/fix-following-gauge.md):
body_state_prediction cut 26-row records into 24-row pieces, the sensor/action contract check built
24-row orders a trunk_v1 world refuses and left the trunk out of its echo check, the causal test's
substitute-order check left the trunk out, and the mastery probe labelled the trunk rows "gaze".
No world is built here.
"""
import types

import numpy as np
import pytest
import torch

from mimo_bridge.body_layout import names_for, names_for_rows


def test_each_row_count_maps_to_its_own_layout():
    assert names_for_rows(24) == names_for("legacy")
    assert names_for_rows(26) == names_for("trunk_v1")
    assert names_for_rows(24)[23] == "gaze" and names_for_rows(26)[23] == "gaze"
    assert names_for_rows(26)[24:] == ("lower_trunk", "chest")
    with pytest.raises(ValueError):
        names_for_rows(25)


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_body_state_prediction_reads_the_wrists_of_the_records_it_is_given(layout):
    from mimo_bridge.tools import body_state_prediction as BSP
    names = list(names_for(layout))
    nb = len(names)
    other = "lower_trunk" if layout == "trunk_v1" else "head"
    for moved, wrist_cm in (("wrist_R", 12.91), (other, 0.0)):
        # 4 states x 4 candidate orders; ONE body point moves, 10 cm more per candidate, along x.
        # The 6 candidate pairs differ by 10, 20, 30, 10, 20, 10 cm on that point: over the two
        # wrists the root mean square is sqrt(2000 / 12) = 12.91 cm when it is a wrist, else 0.
        y = torch.zeros(4, 4, nb * 3, dtype=torch.float64)
        for a in range(4):
            y[:, a, names.index(moved) * 3] = 10.0 * a
        de = BSP.effects(y)
        assert BSP.body_points(de).shape[1:] == (nb, 3)
        assert BSP.wrist_rows(de) == [names.index("wrist_L"), names.index("wrist_R")]
        got = float(BSP.body_points(de)[:, BSP.wrist_rows(de)].square().sum(-1).mean().sqrt())
        assert got == pytest.approx(wrist_cm, abs=0.01), (moved, got)


@pytest.mark.parametrize("n", [24, 26])
def test_the_hold_order_has_one_row_per_body_row(n):
    from mimo_bridge.tools.sensor_action_contract_check import hold_current
    bones = np.zeros((n, 11))
    bones[:, 3] = 1.0
    bones[:, 4:7] = np.arange(n)[:, None] + np.array([0.0, 0.5, 0.25])
    cmd = hold_current(bones)
    assert cmd.shape == (1, n, 9)
    assert np.allclose(cmd[0, :, 0:3], bones[:, 4:7]) and np.allclose(cmd[0, :, 3:7], bones[:, 0:4])


@pytest.mark.parametrize("n", [24, 26])
def test_the_echo_check_covers_every_body_row_and_leaves_out_only_the_eyes(n):
    from mimo_bridge.tools.sensor_action_contract_check import echo_learner_check
    names = list(names_for_rows(n))
    bones = np.zeros((n, 11), np.float32)
    bones[:, 3] = 1.0
    bones[:, 4:7] = 20.0

    def frame(mh):
        return types.SimpleNamespace(tokenized=types.SimpleNamespace(motor_handles=mh), bones=bones)

    good = np.zeros((n, 10), np.float32)
    good[:, 0:3] = bones[:, 4:7] / 200.0
    good[:, 3:7] = bones[:, 0:4]
    assert echo_learner_check(frame(good), 200.0) == (True, True)
    for i, name in enumerate(names):
        bad = good.copy()
        bad[i, 0:3] += 0.5                      # 100 cm off
        bad[i, 3:7] = [1.0, 0.0, 0.0, 0.0]      # a half turn off
        want = (True, True) if name == "gaze" else (False, False)
        assert echo_learner_check(frame(bad), 200.0) == want, name
    assert echo_learner_check(frame(good[:24]), 200.0) == ((True, True) if n == 24 else (False, False))


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_a_substitute_order_counts_as_different_in_any_body_row_but_not_in_the_eyes_row(layout):
    from mimo_bridge.tools.causal_test import body_rows_differ
    names = list(names_for(layout))
    a = torch.zeros(1, len(names), 9)
    a[..., 6] = 1.0
    for i, name in enumerate(names):
        c = a.clone()
        c[:, i, 3:7] = torch.tensor([0.0, 0.0, 0.7071, 0.7071])
        assert body_rows_differ(c, a, names.index("gaze")) == (name != "gaze"), name


def test_the_contract_self_test_drives_every_fake_in_both_body_layouts(capsys):
    # the command-echo fake echoes the order it is sent as its achieved pose; with a 24-row
    # hold order a 26-row (trunk_v1) fake cannot even be stepped
    from mimo_bridge.tools import sensor_action_contract_check as S
    fake = S.CommandEchoFake(26)
    out = S.run_contracts(fake, fake.patch_names, fake.n_patches, settle_steps=10, phase_steps=10,
                          run_name="echo_26")
    assert fake.last_cmd.shape == (26, 9)
    assert out["contracts"]["action"]["achieved_is_command_echo"] is True
    assert out["contracts"]["action"]["pass"] is False
    assert S.self_test() == 0
    assert "26-row trunk_v1 body" in capsys.readouterr().out
