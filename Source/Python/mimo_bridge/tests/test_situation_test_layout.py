# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The situation-dependence test builds the world the checkpoint trained in, body layout included.

What was wrong (found 2026-09-23, fix-rows-r1-situation_test.py): tools/situation_test.py built
its probe world by hand from a dozen args.json keys and never passed body_layout, so the world
was the env's default 24-row legacy body. Every H84 segment trains the trunk_v1 body (26 rows:
the legacy 24 plus lower trunk and chest), whose proprioception encoder takes 26 x 12 = 312
numbers against the legacy world's 288, so the first model step on an H84 checkpoint raised and
situation_test.json could not be refreshed for the H84 lineage. The same hand-built world also
had the legacy contact profile, no body function, synthetic audio, no ring camera and a 5-step
order-smoothing horizon, where the run trains with its own.

Both tests run situation_test.run itself with the model loader replaced; the world is caught the
moment run() builds it. The first compares the world settings with the trainer's own
train_mimo.env_kwargs_for. The second builds the real world (crib, vision off, no caregiver)
from run()'s settings under both layouts and gives it one order of the model's row count.
"""
import types

import numpy as np
import pytest

import mimo_bridge.datasource as datasource
from mimo_bridge import train_mimo
from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import causal_test
from mimo_bridge.tools import situation_test as ST

LIVE_RECORDINGS = "C:/live_run/caregiver_records/H99"
# the situation test's own settings; every other world setting must be the trainer's
OWN = {"run_dir", "vision", "seed", "tag", "caregiver_mode", "caregiver_seconds",
       "caregiver_record_dir", "caregiver_replay", "critters"}
LIMITS = {"phase_name": "mature", "rom_limit": 1.0, "strength_limit": 1.0,
          "age_months": 24.0, "exploration_noise": 0.0}


def _trainer_args(tmp_path, layout):
    """An H84-shaped trainer namespace, as causal_test._build_model returns it (a dict)."""
    ns = train_mimo.parse_args(["--out", str(tmp_path / "run")])
    vars(ns).update(body_layout=layout, contact_profile="mechanical_v4", age_binding="fixed:24",
                    caregiver_mode="responsive", caregiver_seconds=15.0,
                    caregiver_record_dir=LIVE_RECORDINGS, mobile="on", critters="on",
                    gaze_reflex="on", spontaneous_saccades="on", decoder_kp=6.0,
                    audio="real", toy_hand_off="follow", body_function="modulate",
                    ring_fov=150.0, render_every=2, action_repeat=5, motor_hold=5)
    return dict(vars(ns))


class _Built(Exception):
    pass


class _Model:
    def load_checkpoint(self, path, load_optimizer=False):
        pass


def _run_until_the_world(tmp_path, monkeypatch, layout, world):
    """situation_test.run on a stand-in model; `world` receives the world's settings and
    must raise _Built (run() is stopped there, before any model step)."""
    a = _trainer_args(tmp_path, layout)
    monkeypatch.setattr(causal_test, "_build_model",
                        lambda ckpt, device, seed=0, intention=False: (_Model(), a, "args.json", True))
    monkeypatch.setattr(causal_test, "_limits_for_run", lambda ckpt, model: (dict(LIMITS), "test"))
    monkeypatch.setattr(datasource, "SingleEnvSource", world)
    ckpt = str(tmp_path / "run" / "checkpoints" / "ckpt_step_1")
    with pytest.raises(_Built):
        ST.run(ckpt, n_pairs=1, out_json=str(tmp_path / "out.json"), run_dir=str(tmp_path / "probe"))
    return a


@pytest.mark.parametrize("layout", ["trunk_v1", "legacy"])
def test_run_builds_the_trained_world(tmp_path, monkeypatch, layout):
    seen = []

    def spy(**kw):
        seen.append(kw)
        raise _Built

    a = _run_until_the_world(tmp_path, monkeypatch, layout, spy)
    (kw,) = seen
    trainer = train_mimo.env_kwargs_for(types.SimpleNamespace(**a), "mature", str(tmp_path / "t"))
    assert kw.get("body_layout") == layout
    assert kw.get("contact_profile") == "mechanical_v4"
    assert kw.get("body_function") == "modulate"
    assert kw.get("cmd_horizon") == 25          # action_repeat 5 x motor_hold 5, as trained
    differ = {k: (kw.get(k), v) for k, v in trainer.items() if k not in OWN and kw.get(k) != v}
    assert not differ, f"probe world differs from the trainer's: {differ}"
    # the test teleports the ball itself: nothing else may move it or the view
    assert (kw["caregiver_mode"], kw["caregiver_seconds"], kw["critters"]) == ("off", 0.0, False)
    # the probe never writes into the run's live caregiver recordings
    assert kw["caregiver_record_dir"] == "" and kw["caregiver_replay"] == ""
    assert (kw["vision"], kw["seed"], kw["tag"]) == (True, 71, "_sit")
    assert kw["age"] == 24.0


@pytest.mark.parametrize("layout", ["trunk_v1", "legacy"])
def test_the_world_run_builds_takes_the_models_orders(tmp_path, monkeypatch, layout):
    real = datasource.SingleEnvSource
    got = {}

    def build(**kw):
        # vision and audio off only to keep the test light; the rows do not depend on them
        src = real(**dict(kw, vision=False, audio_real=False))
        try:
            frames = src.reset()
            got["rows"] = src.n_handles
            got["names"] = list(src.env.proprio.handle_names)
            got["proprio_numbers"] = int(frames[0].tokenized.proprio.numel())
            rows = len(names_for(layout))           # the model's own row count
            still = np.zeros((1, rows, 9), np.float32)
            still[..., 6] = 1.0                     # identity rotation, zero strength
            src.step(still, np.zeros((1, 2)), None, render=False)
            got["stepped"] = True
        finally:
            src.close()
        raise _Built

    _run_until_the_world(tmp_path, monkeypatch, layout, build)
    names = list(names_for(layout))
    assert got["rows"] == len(names)
    assert got["names"] == names
    assert got["proprio_numbers"] == 12 * len(names)   # 312 trunk_v1, 288 legacy
    assert got.get("stepped")
