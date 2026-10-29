# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The rubbing recalibration's launch flag from the trainer to a world, a resumed world and the model's
own pain formula (worlds, torch). The rule itself is tested without a world in test_rubbing_calibration.py.

Default 'v4' keeps every world exactly as before and adds no key; '_shadow' keeps pain on today's memory
and reports the recalibrated pain beside it; 'excess_shear_v1' feeds pain from the recalibrated memory and
reports today's. Sensing (the touch the model receives, the local function) never changes.
"""
import copy
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge import continuation_state as CS

TRAIN = Path(__file__).resolve().parents[1] / "train_mimo.py"


def _episode(n=2, **kw):
    worlds = []
    for i in range(n):
        env_kwargs = {k: v for k, v in dict(dict(scene="crib", seed=7 + i, tag="_w%d" % i, run_dir="old/scenes",
                                                 contact_profile="mechanical_v4"), **kw).items() if v is not None}
        worlds.append(dict(env_kwargs=env_kwargs, world={}, payload=dict(sim_time=1.0)))
    return dict(version=CS.VERSION, worlds=worlds, config=dict(n_envs=n), trainer=dict(step=10))


def test_the_trainer_flag_defaults_to_v4_and_names_the_world_setting_only_when_changed():
    src = TRAIN.read_text(encoding="utf-8")
    flag = re.search(r"--rubbing-calibration['\"].{0,160}?default=['\"](\w+)['\"]", src, re.S)
    assert flag and flag.group(1) == "v4"
    from mimo_bridge import train_mimo as TM
    args = TM.parse_args(["--out", "unused"])
    assert args.rubbing_calibration == "v4" and "rubbing_calibration" not in TM.env_kwargs_for(args, None, "out")
    for value in ("excess_shear_v1_shadow", "excess_shear_v1"):
        args = TM.parse_args(["--out", "unused", "--contact-profile", "mechanical_v4", "--rubbing-calibration", value])
        assert TM.env_kwargs_for(args, None, "out")["rubbing_calibration"] == value


def test_a_resume_takes_the_calibration_only_when_asked_and_never_at_the_default(caplog):
    state = _episode()
    before = copy.deepcopy(state)
    for mode in ("launch", "saved"):
        up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(rubbing_calibration="v4"), env_settings=mode,
                                            launch_kwargs=dict(state["worlds"][0]["env_kwargs"]))
        assert up is state and changes == [] and state == before
    launch = SimpleNamespace(rubbing_calibration="excess_shear_v1")
    with caplog.at_level(logging.INFO, logger="mimo_bridge.continuation_state"):
        kept, none = CS.migrate_env_kwargs(state, launch)
    assert kept is state and none == []
    assert any("rubbing_calibration" in r.getMessage() and "NOT applied" in r.getMessage() for r in caplog.records)
    up, changes = CS.migrate_env_kwargs(state, launch, env_settings="launch")
    assert [c["new"] for c in changes] == ["excess_shear_v1"] * 2 and state == before
    assert all(w["env_kwargs"]["rubbing_calibration"] == "excess_shear_v1" for w in up["worlds"])
    with pytest.raises(ValueError, match="rubbing_calibration"):
        CS.migrate_env_kwargs(state, SimpleNamespace(rubbing_calibration="soft"), env_settings="launch")


def test_the_env_refuses_an_unknown_calibration_and_one_its_skin_cannot_use(tmp_path):
    from mimo_bridge.env import BridgeEnv
    with pytest.raises(ValueError, match="unknown rubbing calibration"):
        BridgeEnv(scene="crib", vision=False, run_dir=str(tmp_path), contact_profile="mechanical_v4",
                  rubbing_calibration="soft")
    with pytest.raises(ValueError, match="virtual skin"):
        BridgeEnv(scene="crib", vision=False, run_dir=str(tmp_path), contact_profile="mechanical_v2",
                  rubbing_calibration="excess_shear_v1")


def _hold(frame):
    command = np.zeros((1, len(frame.bones), 9), np.float32)
    command[0, :, :3] = frame.bones[:, 4:7]
    command[0, :, 3:7] = frame.bones[:, :4]
    return command


def _model_pain(frame, sensitivity):
    """The model's pain on this frame: contact_response on the frame's pressure and history (worst patch)."""
    import torch
    from lem.model.contact_wellness import contact_response
    patches = np.asarray(frame.patches, dtype=np.float64)
    f = torch.zeros(1, len(patches), 8, dtype=torch.float64)
    f[0, :, 0] = torch.tensor(patches[:, 0] / 1e5)
    gains = torch.ones(1, len(patches), 2, dtype=torch.float64)
    gains[0, :, 0] = torch.tensor(np.asarray(sensitivity, dtype=np.float64)[:, 0])
    history = torch.tensor(np.asarray(frame.meta["mechanical_contact"]["history"], dtype=np.float64))[None]
    return float(contact_response(f, gains, profile="mechanical_v4", history=history)["pain"])


KW = dict(scene="crib", vision=False, age=24, caregiver_seconds=0, body_function="modulate",
          contact_profile="mechanical_v4")


@pytest.mark.slow
def test_each_setting_feeds_pain_from_its_memory_and_never_changes_sensing(tmp_path):
    """Three crib worlds with the same seed and the same orders: 'v4', shadow and on. Physics, the touch the
    model receives and the local function are identical in all three. Shadow's history is v4's; the report's
    pain for the rule in force equals the model's own formula on the frame; on feeds the recalibrated memory."""
    from mimo_bridge.datasource import SingleEnvSource
    sources = {s: SingleEnvSource(**dict(KW, run_dir=str(tmp_path / s), **({} if s == "v4" else {"rubbing_calibration": s})))
               for s in ("v4", "excess_shear_v1_shadow", "excess_shear_v1")}
    try:
        frames = {s: src.reset()[0] for s, src in sources.items()}
        command = _hold(frames["v4"])
        for _ in range(12):
            frames = {s: src.step(command)[0] for s, src in sources.items()}
        v4, shadow, on = frames["v4"], frames["excess_shear_v1_shadow"], frames["excess_shear_v1"]
        for other in (shadow, on):
            np.testing.assert_array_equal(other.bones, v4.bones)
            np.testing.assert_array_equal(other.patches, v4.patches)
            np.testing.assert_array_equal(other.meta["body_function"]["values"], v4.meta["body_function"]["values"])
        assert "rubbing_calibration" not in v4.meta["mechanical_contact"]
        np.testing.assert_array_equal(shadow.meta["mechanical_contact"]["history"], v4.meta["mechanical_contact"]["history"])
        sens = sources["v4"].env.touch_sensitivity
        for frame, rule in ((shadow, "v4"), (on, "excess_shear_v1")):
            report = frame.meta["mechanical_contact"]["rubbing_calibration"]
            assert report["pain_rule"] == rule and set(report["pain"]) == {"v4", "excess_shear_v1"}
            assert report["pain"][rule] == pytest.approx(_model_pain(frame, sens), abs=1e-6)
            rules = report["since_built"]["rules"]
            assert sum(rules["v4"]["histogram"]) == sum(rules["excess_shear_v1"]["histogram"]) \
                == report["since_built"]["frames"]
            for r in rules.values():
                assert sum(r["ordinary_histogram"]) <= sum(r["histogram"])
        # the same two rules on the same physics read the same in both worlds
        assert shadow.meta["mechanical_contact"]["rubbing_calibration"]["pain"] == \
            on.meta["mechanical_contact"]["rubbing_calibration"]["pain"]
        skin = sources["excess_shear_v1"].env._virtual_skin
        on_history = np.asarray(on.meta["mechanical_contact"]["history"])
        assert (on_history[:, 2] <= np.asarray(v4.meta["mechanical_contact"]["history"])[:, 2] + 1e-9).all()
        np.testing.assert_array_equal(on_history[:, :2], np.asarray(v4.meta["mechanical_contact"]["history"])[:, :2])
        assert skin.pain_exposure is not None and sources["v4"].env._virtual_skin.pain_exposure is None
    finally:
        for src in sources.values():
            src.close()


@pytest.mark.slow
def test_a_saved_v4_world_resumes_onto_the_calibration_and_a_calibrated_world_restores_exactly(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    source = SingleEnvSource(**dict(KW, run_dir=str(tmp_path / "a")))
    restored = None
    try:
        frames = source.reset()
        command = _hold(frames[0])
        for _ in range(6):
            frames = source.step(command)
        saved = dict(version=CS.VERSION, worlds=source.save_full_state(), config={}, trainer=dict(step=6))
        stored = np.array(source.env._virtual_skin.exposure.values)
        up, changes = CS.migrate_env_kwargs(saved, SimpleNamespace(rubbing_calibration="excess_shear_v1"),
                                            env_settings="launch")
        assert len(changes) == 1
        restored = SingleEnvSource(**up["worlds"][0]["env_kwargs"])
        restored.restore_full_state(up["worlds"])
        skin = restored.env._virtual_skin
        np.testing.assert_array_equal(skin.exposure.values, stored)
        np.testing.assert_array_equal(skin.pain_exposure.values, stored)     # starts from the stored loads
        expected = source.step(command)[0]
        actual = restored.step(command)[0]
        np.testing.assert_array_equal(actual.bones, expected.bones)
        np.testing.assert_array_equal(actual.patches, expected.patches)
        # the calibrated world saves its second memory and restores exactly
        snapshot = restored.save_full_state()
        assert snapshot[0]["env_kwargs"]["rubbing_calibration"] == "excess_shear_v1"
        nxt = restored.step(command)[0]
        restored.restore_full_state(snapshot)
        again = restored.step(command)[0]
        np.testing.assert_array_equal(again.patches, nxt.patches)
        np.testing.assert_array_equal(again.meta["mechanical_contact"]["history"], nxt.meta["mechanical_contact"]["history"])
        # and a v4 world built from that calibrated state drops the second memory
        v4 = SingleEnvSource(**dict(snapshot[0]["env_kwargs"], rubbing_calibration="v4"))
        try:
            v4.restore_full_state([dict(snapshot[0], env_kwargs=v4.env_kwargs)])
            assert v4.env._virtual_skin.pain_exposure is None
        finally:
            v4.close()
    finally:
        source.close()
        if restored is not None:
            restored.close()
