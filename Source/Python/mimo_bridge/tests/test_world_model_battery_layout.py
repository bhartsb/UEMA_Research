# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The world-model battery builds the world the checkpoint trained in, body layout included.

What was wrong (found 2026-09-23, fix-rows-r1-world_model_battery.py): both parts of
tools/world_model_battery.py built their probe world by hand from a dozen args.json keys and
never passed body_layout or contact_profile, so the world was the env's default 24-row legacy
body. Every H84 segment trains the trunk_v1 body (26 rows: the legacy 24 plus lower trunk and
chest), whose proprioception encoder takes 26 x 12 = 312 numbers against the legacy world's 288,
so the first model step on an H84 checkpoint raised. Part C also sent a fixed 24-row "hold still"
order, which a 26-row world refuses.

The first two tests replace the world and the model with stand-ins that refuse a mismatch the
way the real ones do (the source's reshape to its own row count; the encoder's input width). The
third builds the real world (crib, vision off, no caregiver) from the battery's kwargs under
both layouts.
"""
import types

import numpy as np
import pytest
import torch

import mimo_bridge.datasource as datasource
from mimo_bridge import train_mimo
from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import world_model_battery as wmb

LIVE_RECORDINGS = "C:/live_run/caregiver_records/H99"
# the battery's own settings in each part; every other world setting must be the trainer's
OWN_BOTH = {"scene", "run_dir", "seed", "tag", "caregiver_record_dir", "caregiver_replay"}
OWN_C = OWN_BOTH | {"gaze_reflex", "spontaneous_saccades", "mobile", "critters",
                    "caregiver_mode", "caregiver_seconds"}


def _trainer_args(tmp_path, layout):
    """An H84-shaped trainer namespace, as causal_test._build_model returns it (a dict)."""
    ns = train_mimo.parse_args(["--out", str(tmp_path / "run")])
    vars(ns).update(body_layout=layout, contact_profile="mechanical_v4", age_binding="fixed:24",
                    caregiver_mode="responsive", caregiver_seconds=15.0,
                    caregiver_record_dir=LIVE_RECORDINGS, mobile="on", critters="on",
                    gaze_reflex="on", spontaneous_saccades="on", decoder_kp=6.0,
                    audio="real", toy_hand_off="follow", body_function="modulate")
    return dict(vars(ns))


LIMITS = {"phase_name": "mature", "rom_limit": 1.0, "strength_limit": 1.0,
          "age_months": 24.0, "exploration_noise": 0.0}
BATTERY_ARGS = types.SimpleNamespace(scene="crib", n=3, device="cpu", away_orders=[2])


class _Built(Exception):
    pass


@pytest.mark.parametrize("layout", ["trunk_v1", "legacy"])
def test_both_parts_build_the_trained_world(tmp_path, monkeypatch, layout):
    a = _trainer_args(tmp_path, layout)
    seen = []

    def spy(**kw):
        seen.append(kw)
        raise _Built

    monkeypatch.setattr(datasource, "SingleEnvSource", spy)
    with pytest.raises(_Built):
        wmb.part_a_and_b(None, BATTERY_ARGS, a, dict(LIMITS))
    with pytest.raises(_Built):
        wmb.part_c(None, BATTERY_ARGS, a, dict(LIMITS))
    ab, c = seen
    trainer = train_mimo.env_kwargs_for(types.SimpleNamespace(**a), "mature", str(tmp_path / "t"))
    for kw, own in ((ab, OWN_BOTH), (c, OWN_C)):
        assert kw.get("body_layout") == layout
        assert kw.get("contact_profile") == "mechanical_v4"
        # the probe never writes into the run's live caregiver recordings
        assert kw["caregiver_record_dir"] == "" and kw["caregiver_replay"] == ""
        differ = {k: (kw.get(k), v) for k, v in trainer.items() if k not in own and kw.get(k) != v}
        assert not differ, f"probe world differs from the trainer's: {differ}"
    assert ab["caregiver_mode"] == "responsive" and ab["mobile"] and ab["critters"]
    # part C: the eyes go only where the plan sends them, nothing moves on its own
    assert (c["gaze_reflex"], c["spontaneous_saccades"], c["mobile"], c["critters"],
            c["caregiver_mode"], c["caregiver_seconds"]) == (False, False, False, False, "off", 0.0)
    assert ab["age"] == 24.0 and c["age"] == 24.0


class _Frame:
    def __init__(self, rows):
        self.rows = rows


class _World:
    """Stands in for SingleEnvSource: its row count is the env's for the layout it is given
    (legacy when none is given, the env's default), and it reshapes an order to its own rows
    exactly as SingleEnvSource.step does."""

    def __init__(self, **kw):
        self.n_handles = len(names_for(kw.get("body_layout", "legacy")))

    def reset(self):
        return [_Frame(self.n_handles)]

    def step(self, motor, gaze=None, handle_mask=None, render=None):
        np.asarray(motor).reshape(-1, self.n_handles, 9)
        return [_Frame(self.n_handles)]

    def close(self):
        pass


class _Model:
    """Stands in for the model: its proprioception encoder takes rows x 12 numbers for ITS
    layout and refuses any other frame, as the real Linear does."""

    def __init__(self, layout):
        self.n_rows = len(names_for(layout))
        self.g = torch.Generator().manual_seed(0)

    def reset_transition_state(self):
        pass

    def step(self, frames, dev, train=False, step_index=0):
        if frames[0].rows != self.n_rows:
            raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied "
                               f"(1x{frames[0].rows * 12} and {self.n_rows * 12}x192)")
        self.last_tokens = {"vision": torch.randn(8, generator=self.g)}
        self.last_predictions = {"vision": torch.randn(8, generator=self.g)}


@pytest.mark.parametrize("layout", ["trunk_v1", "legacy"])
def test_look_back_runs_on_the_models_own_body(tmp_path, monkeypatch, layout):
    monkeypatch.setattr(datasource, "SingleEnvSource", _World)
    out = wmb.part_c(_Model(layout), BATTERY_ARGS, _trainer_args(tmp_path, layout), dict(LIMITS))
    assert set(out) == {"2"}
    assert np.isfinite(out["2"]["look_back_advantage"])


class _BodyModel:
    """Enough of LemLite for causal_test._collect. Its 'tokens' are the world's raw body poses
    and skin readings, so a collected record carries the physical state the world reported, and
    its order moves every row at 0.3 strength."""

    def __init__(self, layout):
        self.n_rows = len(names_for(layout))
        self.ctx_len, self.n_mod, self.d_model = 3, 2, 4
        self.context_buffer = None

    def reset_transition_state(self):
        self.context_buffer = None
        self._nf_ctx = None

    def step(self, frames, dev, train, step_index, executed_action, executed_order_id):
        shape = (1, (self.ctx_len - 1) * self.n_mod, self.d_model)
        self.context_buffer = torch.zeros(shape)
        self._nf_ctx = torch.zeros(shape)
        self.last_tokens = {"bones": torch.as_tensor(np.array(frames[0].bones, dtype=np.float64)),
                            "patches": torch.as_tensor(np.array(frames[0].patches, dtype=np.float64))}
        self.last_achieved = torch.zeros(1, self.n_rows, 10)
        order = np.zeros((1, self.n_rows, 9), np.float32)
        order[..., 3:7] = (0.0, 0.2, 0.0, 0.98)        # a turn, so the two arms differ
        order[..., 8] = 0.3
        return types.SimpleNamespace(motor_for_action=order, gaze_out=np.zeros((1, 2), np.float32))


class _NoBranch:
    """The same world with its env hidden, so _collect runs no passive branch in it."""
    env = None

    def __init__(self, src):
        self.src = src

    def reset(self):
        return self.src.reset()

    def step(self, *args, **kw):
        return self.src.step(*args, **kw)


@pytest.mark.parametrize("layout", ["trunk_v1", "legacy"])
def test_the_passive_branch_leaves_the_collected_stream_unchanged(tmp_path, layout):
    """Part A/B collect through causal_test._collect, which runs every order twice from the
    same state: once with no action (the passive branch) and once for real. The trainer's world
    has a skin with its own clock (contact_profile mechanical_v4) and a responsive caregiver;
    the branch put the physics back but not those, so the real arm stopped on 'virtual skin time
    reversed'. With every stateful part put back, the real arm is the stream a world with no
    branch collects."""
    from mimo_bridge.tools import causal_test as CT
    kw = wmb.probe_world_kwargs(_trainer_args(tmp_path, layout), dict(LIMITS), str(tmp_path / "w"),
                                scene="crib", seed=140, tag=f"_wmbp_{layout}",
                                overrides=dict(vision=False))
    limits = dict(LIMITS, _action_repeat=2)
    streams = []
    for branch in (True, False):
        src = datasource.SingleEnvSource(**kw)
        try:
            streams.append(CT._collect(_BodyModel(layout), src if branch else _NoBranch(src), 4, 2,
                                       limits=limits))
        finally:
            src.close()
    with_branch, without = streams
    assert len(with_branch) == len(without) == 3
    assert all("passive_delta" in r for r in with_branch)
    for a, b in zip(with_branch, without):
        for key in ("bones", "patches"):
            assert torch.equal(a["t1"][key], b["t1"][key]), key


@pytest.mark.parametrize("layout", ["trunk_v1", "legacy"])
def test_the_battery_world_is_built_with_the_trained_rows(tmp_path, layout):
    kw = wmb.probe_world_kwargs(_trainer_args(tmp_path, layout), dict(LIMITS), str(tmp_path / "w"),
                                scene="crib", seed=140, tag=f"_wmbt_{layout}",
                                overrides=dict(vision=False, caregiver_mode="off",
                                               caregiver_seconds=0.0, audio_real=False))
    src = datasource.SingleEnvSource(**kw)
    try:
        assert src.n_handles == len(names_for(layout))
        assert src.env.body_layout == layout
        assert src.env.contact_profile == "mechanical_v4"
        assert list(src.env.proprio.handle_names) == list(names_for(layout))
    finally:
        src.close()
