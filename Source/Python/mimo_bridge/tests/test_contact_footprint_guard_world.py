# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The footprint guard's launch flag, from the trainer's argument to a resumed world (worlds, torch).

Default 'off' keeps every world exactly as before; a resume switches a saved world onto the guard
only under --resume-env-settings launch, and the world keeps its skin's saved memories. The guard
changes how contact is scored, never the physics. The footprint itself is tested without a world in
test_contact_footprint_guard.py.
"""
import copy
import logging
from pathlib import Path
import re
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


# ------------------------------------------------------------------ the flag and its migration
def test_the_trainer_flag_defaults_off_and_reaches_the_world_settings_only_when_on():
    src = TRAIN.read_text(encoding="utf-8")
    flag = re.search(r"--contact-footprint-guard['\"].{0,120}?default=['\"](\w+)['\"]", src, re.S)
    assert flag and flag.group(1) == "off"
    from mimo_bridge import train_mimo as TM
    args = TM.parse_args(["--out", "unused"])
    assert args.contact_footprint_guard == "off"
    # at the default the world settings carry no key: exactly the settings older code builds
    assert "contact_footprint_guard" not in TM.env_kwargs_for(args, None, "out")
    args = TM.parse_args(["--out", "unused", "--contact-profile", "mechanical_v4",
                          "--contact-footprint-guard", "rounded_edge_v1"])
    assert TM.env_kwargs_for(args, None, "out")["contact_footprint_guard"] == "rounded_edge_v1"


def test_a_resume_at_the_default_adds_no_key_and_names_nothing(caplog):
    """H84 resumes with --resume-env-settings launch: a plain resume of its command on this code must
    leave every saved world's settings byte-identical, or code without the setting could not build
    the next checkpoint's worlds."""
    state = _episode()
    before = copy.deepcopy(state)
    with caplog.at_level(logging.INFO, logger="mimo_bridge.continuation_state"):
        for mode in ("launch", "saved"):
            up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(contact_footprint_guard="off"),
                                                env_settings=mode, launch_kwargs=dict(state["worlds"][0]["env_kwargs"]))
            assert up is state and changes == [] and state == before, mode
    assert not [r for r in caplog.records if "contact_footprint_guard" in r.getMessage()]
    on = _episode(contact_footprint_guard="rounded_edge_v1")
    up, changes = CS.migrate_env_kwargs(on, SimpleNamespace(contact_footprint_guard="off"), env_settings="launch")
    assert [c["old"] for c in changes] == ["rounded_edge_v1", "rounded_edge_v1"], "switching back off is a change"
    assert all(w["env_kwargs"]["contact_footprint_guard"] == "off" for w in up["worlds"])


def test_a_resume_switches_the_guard_on_only_when_asked_and_never_edits_the_saved_state(caplog):
    state = _episode()                                   # a companion from before the flag existed
    before = copy.deepcopy(state)
    launch = SimpleNamespace(contact_footprint_guard="rounded_edge_v1")
    with caplog.at_level(logging.INFO, logger="mimo_bridge.continuation_state"):
        kept, none = CS.migrate_env_kwargs(state, launch)                       # the default: saved
    assert kept is state and none == [] and state == before
    assert any("contact_footprint_guard" in r.getMessage() and "NOT applied" in r.getMessage()
               for r in caplog.records)
    up, changes = CS.migrate_env_kwargs(state, launch, env_settings="launch")
    assert [(c["world"], c["old"], c["new"]) for c in changes] == [(0, None, "rounded_edge_v1"),
                                                                    (1, None, "rounded_edge_v1")]
    assert all(w["env_kwargs"]["contact_footprint_guard"] == "rounded_edge_v1" for w in up["worlds"])
    assert up["config"] is state["config"] and state == before
    with pytest.raises(ValueError, match="contact_footprint_guard"):
        CS.migrate_env_kwargs(state, SimpleNamespace(contact_footprint_guard="sometimes"), env_settings="launch")


def test_the_env_refuses_an_unknown_guard_and_a_guard_the_profile_cannot_use(tmp_path):
    from mimo_bridge.env import BridgeEnv
    with pytest.raises(ValueError, match="unknown contact footprint guard"):
        BridgeEnv(scene="crib", vision=False, run_dir=str(tmp_path), contact_profile="mechanical_v4",
                  contact_footprint_guard="sometimes")
    with pytest.raises(ValueError, match="mechanical_v3 and mechanical_v4"):
        BridgeEnv(scene="crib", vision=False, run_dir=str(tmp_path), contact_profile="mechanical_v2_local",
                  contact_footprint_guard="rounded_edge_v1")


# ------------------------------------------------------------------ worlds
def _hold(frame):
    command = np.zeros((1, len(frame.bones), 9), np.float32)
    command[0, :, :3] = frame.bones[:, 4:7]
    command[0, :, 3:7] = frame.bones[:, :4]
    return command


@pytest.mark.slow
def test_a_saved_world_resumes_onto_the_guard_with_its_skin_memories_and_physics_unchanged(tmp_path):
    """The production switch at 1 body: a mechanical_v4 crib world saved with the guard off (no key in
    its settings, like every H84 companion), migrated with 'launch', rebuilt from the migrated settings
    exactly as train_mimo rebuilds a resumed world, and restored. Its per-taxel memories are the saved
    ones; its next step moves the body exactly as the unguarded world's does (the guard only scores
    contact); the report names the setting. A world built with 'off' spelled out steps identically to
    one built without the setting."""
    from mimo_bridge.datasource import SingleEnvSource
    kw = dict(scene="crib", vision=False, run_dir=str(tmp_path / "a"), age=24, caregiver_seconds=0,
              body_function="modulate", contact_profile="mechanical_v4")
    source = SingleEnvSource(**kw)
    restored = spelled = None
    try:
        frames = source.reset()
        command = _hold(frames[0])
        for _ in range(3):
            frames = source.step(command)
        assert "footprint_guard" not in frames[0].meta["mechanical_contact"]
        saved = dict(version=CS.VERSION, worlds=source.save_full_state(), config={}, trainer=dict(step=3))
        assert "contact_footprint_guard" not in saved["worlds"][0]["env_kwargs"]
        memories = np.array(source.env._virtual_skin.exposure.values)
        up, changes = CS.migrate_env_kwargs(saved, SimpleNamespace(contact_footprint_guard="rounded_edge_v1"),
                                            env_settings="launch")
        assert len(changes) == 1
        restored = SingleEnvSource(**up["worlds"][0]["env_kwargs"])          # train_mimo's resume line
        restored.restore_full_state(up["worlds"])
        assert restored.env.contact_footprint_guard == "rounded_edge_v1"
        np.testing.assert_array_equal(restored.env._virtual_skin.exposure.values, memories)
        expected = source.step(command)[0]
        actual = restored.step(command)[0]
        np.testing.assert_array_equal(actual.bones, expected.bones)
        report = actual.meta["mechanical_contact"]["footprint_guard"]
        assert report["setting"] == "rounded_edge_v1" and report["pressure_ceiling_pa"] == 1e7
        assert set(report["since_built"]) == {"rescored_flat_overlaps", "pressure_ceiling_contacts",
                                              "largest_unguarded_pressure_pa", "rule_violations",
                                              "last_violation"}
        assert report["since_built"]["rule_violations"] == 0 and report["since_built"]["last_violation"] is None
        assert restored.save_full_state()[0]["env_kwargs"]["contact_footprint_guard"] == "rounded_edge_v1"
        # 'off' spelled out is the world without the setting
        spelled = SingleEnvSource(**dict(kw, run_dir=str(tmp_path / "b"), contact_footprint_guard="off"))
        plain = SingleEnvSource(**dict(kw, run_dir=str(tmp_path / "c")))
        try:
            a, b = spelled.reset()[0], plain.reset()[0]
            for _ in range(3):
                a, b = spelled.step(_hold(a))[0], plain.step(_hold(b))[0]
            np.testing.assert_array_equal(a.patches, b.patches)
            np.testing.assert_array_equal(a.bones, b.bones)
            assert a.meta["mechanical_contact"] == b.meta["mechanical_contact"]
        finally:
            plain.close()
    finally:
        source.close()
        for s in (restored, spelled):
            if s is not None:
                s.close()


@pytest.mark.slow
def test_a_guarded_world_saves_and_restores_exactly(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    source = SingleEnvSource(scene="crib", vision=False, run_dir=str(tmp_path), age=24, caregiver_seconds=0,
                             body_function="modulate", contact_profile="mechanical_v4",
                             contact_footprint_guard="rounded_edge_v1")
    try:
        frames = source.reset()
        command = _hold(frames[0])
        for _ in range(3):
            frames = source.step(command)
        snapshot = source.save_full_state()
        expected = source.step(command)[0]
        source.restore_full_state(snapshot)
        actual = source.step(command)[0]
        np.testing.assert_array_equal(actual.patches, expected.patches)
        np.testing.assert_array_equal(actual.bones, expected.bones)
        np.testing.assert_array_equal(actual.meta["body_function"]["values"], expected.meta["body_function"]["values"])
    finally:
        source.close()
