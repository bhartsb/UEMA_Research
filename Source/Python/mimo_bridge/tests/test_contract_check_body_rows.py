# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The sensor/action contract check (spec 20.6) runs in both body layouts and checks every body row.

Every H84 segment runs the trunk_v1 layout: 26 body rows, where rows 24 and 25 (lower trunk,
chest) come AFTER the eyes' row (row 23). The legacy layout has 24 rows and ends with the eyes.
Until 2026-09-23 tools/sensor_action_contract_check.py assumed 24 rows:
  - hold_current built a 24-row order, which a 26-row world refuses on its first step (the
    reshape in datasource.SingleEnvSource.step); first_action_benefit, consequence_localization
    and action_effect_learning build their hold orders with the same function;
  - echo_learner_check compared rows 0-22 only, so a learner motor tensor with the lower trunk
    and the chest wrong, or with rows missing, still passed;
  - its negative-control fakes had 24 rows, so the qualification that runs before every assay
    never saw the layout the assay faces.
Layout names come from body_layout.names_for. No world is built here.
"""
import types

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import sensor_action_contract_check as C

LAYOUTS = ("legacy", "trunk_v1")
RNG_CM = 200.0      # tokenizer position_range_cm: the learner's motor tensor holds world cm / 200


def _bones(n):
    """n body rows, each at its own place (cm) with the identity rotation (quat xyzw)."""
    b = np.zeros((n, 11))
    b[:, 3] = 1.0
    b[:, 4:7] = np.arange(n)[:, None] * 3.0 + np.array([10.0, 20.0, 30.0])
    return b


def _learner_copy(bones):
    """The learner's motor tensor as the data source writes it: achieved pose, world cm / 200."""
    mh = np.zeros((len(bones), 10), np.float32)
    mh[:, 0:3] = bones[:, 4:7] / RNG_CM
    mh[:, 3:7] = bones[:, 0:4]
    return mh


def _frame(bones, motor_handles):
    return types.SimpleNamespace(bones=bones,
                                 tokenized=types.SimpleNamespace(motor_handles=motor_handles))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_hold_order_has_one_row_per_body_row_and_the_world_accepts_it(layout):
    n = len(names_for(layout))
    bones = _bones(n)
    cmd = C.hold_current(bones)
    assert cmd.shape == (1, n, 9)
    rows = cmd.reshape(-1, n, 9)[0]          # what datasource.SingleEnvSource.step does to it
    assert np.allclose(rows[:, 0:3], bones[:, 4:7]) and np.allclose(rows[:, 3:7], bones[:, 0:4])
    assert np.all(rows[:, 7] == 0.5) and np.all(rows[:, 8] == 1.0)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_echo_check_catches_a_wrong_row_anywhere_in_the_body_but_not_in_the_eyes_row(layout):
    names = list(names_for(layout))
    bones = _bones(len(names))
    good = _learner_copy(bones)
    assert C.echo_learner_check(_frame(bones, good), RNG_CM) == (True, True)
    for i, name in enumerate(names):
        bad = good.copy()
        bad[i, 0:3] += 5.0                      # 1000 cm from where the body part is
        bad[i, 3:7] = [1.0, 0.0, 0.0, 0.0]      # a half turn from its rotation
        want = (True, True) if name == "gaze" else (False, False)
        assert C.echo_learner_check(_frame(bones, bad), RNG_CM) == want, name


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_learner_tensor_with_a_different_row_count_fails_the_echo_check(layout):
    n = len(names_for(layout))
    bones = _bones(n)
    good = _learner_copy(bones)
    # trunk_v1: the 24 rows of the UE byte frame without the trunk; legacy: the eyes' row dropped
    short = good[:24] if layout == "trunk_v1" else good[:23]
    assert C.echo_learner_check(_frame(bones, short), RNG_CM) == (False, False)
    long_ = np.concatenate([good, good[:1]])
    assert C.echo_learner_check(_frame(bones, long_), RNG_CM) == (False, False)


class _SizedSource:
    """A frame source with n body rows that, like datasource.SingleEnvSource.step, refuses an
    order whose row count is not its own. Its learner copy carries the achieved pose on every
    row except `off_row`, which sits 1000 cm from the body part."""
    patch_names = ["right_hand", "left_foot"]
    n_patches = 2

    def __init__(self, n, off_row=None):
        self.n, self.off_row, self.i, self.orders = n, off_row, 0, 0

    def _frame(self):
        b = _bones(self.n)
        mh = _learner_copy(b)
        if self.off_row is not None:
            mh[self.off_row, 0:3] += 5.0
        md = types.SimpleNamespace(frame_number=self.i, timestamp=self.i * 0.01,
                                   gaze_azimuth_deg=0.0, gaze_elevation_deg=0.0)
        tok = types.SimpleNamespace(metadata=md, touch_impulses=np.zeros((256, 8), np.float32),
                                    touch_valid_counts=(2, 1), motor_handles=mh)
        return types.SimpleNamespace(bones=b, patches=np.zeros((2, 10)), n_contacts=0,
                                     frame=self.i, sim_time=self.i * 0.01, gaze_deg=(0.0, 0.0),
                                     tokenized=tok, meta={}, vision_patches=None)

    def reset(self):
        return [self._frame()]

    def step(self, motor, gaze, mask, render=None):
        np.asarray(motor).reshape(-1, self.n, 9)
        self.i += 1
        self.orders += 1
        return [self._frame()]

    def rebuild(self, **changes):
        pass


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_whole_assay_runs_on_a_world_of_either_layout(layout):
    names = list(names_for(layout))
    src = _SizedSource(len(names))
    out = C.run_contracts(src, src.patch_names, src.n_patches, settle_steps=4, phase_steps=4,
                          run_name=layout)
    # settle 4, press 2 x 4, release 4, free 4, then the eye assay: settle 30 + 4 x (40 + 40)
    assert src.orders == 4 + 8 + 4 + 4 + 30 + 4 * 80
    assert out["contracts"]["action"]["learner_motor_is_achieved_at_rest"] is True
    # the last body row that is not the eyes: row 22 (legacy), the chest at row 25 (trunk_v1)
    last_body = max(i for i, nm in enumerate(names) if nm != "gaze")
    off = C.run_contracts(_SizedSource(len(names), off_row=last_body), src.patch_names,
                          src.n_patches, settle_steps=4, phase_steps=4, run_name=layout)
    assert off["contracts"]["action"]["learner_motor_is_achieved_at_rest"] is False


NEGATIVE_CONTROLS = (("StaleUnrelatedFake", ("touch", "action", "vision", "freshness")),
                     ("CollisionOnlyFake", ("touch",)),
                     ("StaleImageFake", ("vision",)),
                     ("CommandEchoFake", ("action",)))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fake_name, must_fail", NEGATIVE_CONTROLS)
def test_every_negative_control_has_the_layouts_rows_and_fails_its_contract(layout, fake_name,
                                                                          must_fail):
    n = len(names_for(layout))
    fake = getattr(C, fake_name)()
    fake.n_rows = n
    assert len(fake.reset()[0].bones) == n
    with pytest.raises(ValueError):             # an order sized for the other layout is refused
        fake.step(np.zeros((1, 50 - n, 9)), np.zeros((1, 2)), None)
    fake = getattr(C, fake_name)()
    fake.n_rows = n
    out = C.run_contracts(fake, fake.patch_names, fake.n_patches, settle_steps=10,
                          phase_steps=10, run_name=fake_name)
    assert not out["all_pass"]
    for c in must_fail:
        assert not out["contracts"][c]["pass"], c


def test_the_self_test_passes_in_both_layouts(capsys):
    assert C.self_test() == 0
    out = capsys.readouterr().out
    assert "24-row" in out and "26-row" in out


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_rows_the_checker_names_are_those_body_parts(layout):
    names = names_for(layout)
    assert names[C.RIGHT_WRIST] == "wrist_R"
    assert names[C.GAZE_HANDLE] == "gaze"
