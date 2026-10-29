# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A setting changed at a resume reaches the rebuilt worlds only when asked for
(continuation_state.migrate_env_kwargs, env_settings='launch').

Measured 2026-09-22: segment 31 was launched with --toy-hand-off follow, and all 161 of its stats
records read object_follows_grip false with 0 grips. A resumed episode rebuilds each world from
the env settings saved in the checkpoint's companion (train_mimo builds SingleEnvSource /
PoolSource from saved_worlds[i]['env_kwargs']), so the resume's flag never reached a world.

DESIGN D2 (a supervisor resume of the old command on new code changes nothing) makes the default
'saved': the worlds keep their settings and every differing launch setting is named as NOT
applied (review ops-storage finding 1, 2026-09-23). The deploy passes 'launch'.
"""
import copy
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge import continuation_state as CS

SPEECH = Path(__file__).resolve().parents[4] / "Saved/mimo_bridge/caregiver_speech_exchange_v1_final"


def _episode(n=2, **kw):
    base = dict(scene="crib", seed=7, tag="_w0", run_dir="old/scenes", toy_hand_off="off",
                finger_armature=0.0004)
    base.update(kw)
    worlds = []
    for i in range(n):
        env_kwargs = {k: v for k, v in dict(base, seed=7 + i, tag="_w%d" % i).items()
                      if v is not None}
        worlds.append(dict(env_kwargs=env_kwargs,
                           world=dict(caregiver=dict(grip=dict(held=False, grips=0, releases=0))),
                           payload=dict(sim_time=1.0)))
    return dict(version=CS.VERSION, worlds=worlds, config=dict(n_envs=n), trainer=dict(step=10))


def test_a_changed_hand_off_is_written_into_every_saved_world_and_nothing_else_changes():
    state = _episode()
    before = copy.deepcopy(state)
    up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(toy_hand_off="follow",
                                                               finger_armature=0.0004),
                                        env_settings="launch")
    assert [(c["world"], c["name"], c["old"], c["new"]) for c in changes] == [
        (0, "toy_hand_off", "off", "follow"), (1, "toy_hand_off", "off", "follow")]
    assert [w["env_kwargs"]["toy_hand_off"] for w in up["worlds"]] == ["follow", "follow"]
    # everything else in each world is the saved one, shared rather than copied
    for old, new in zip(state["worlds"], up["worlds"]):
        assert new["world"] is old["world"] and new["payload"] is old["payload"]
        assert {k: v for k, v in new["env_kwargs"].items() if k != "toy_hand_off"} == \
            {k: v for k, v in old["env_kwargs"].items() if k != "toy_hand_off"}
    assert up["config"] is state["config"], "the continuation config must not change"
    assert state == before, "the saved state itself was edited"


def test_nothing_to_change_returns_the_same_state_and_no_changes():
    state = _episode(toy_hand_off="follow")
    up, changes = CS.migrate_env_kwargs(state, dict(toy_hand_off="follow", finger_armature=0.0004),
                                        env_settings="launch")
    assert up is state and changes == []


def test_a_setting_the_saved_world_never_carried_is_added_and_reported_as_absent():
    state = _episode(toy_hand_off=None)                 # a companion from before the flag existed
    up, changes = CS.migrate_env_kwargs(state, dict(toy_hand_off="off"), env_settings="launch")
    assert {c["old"] for c in changes} == {None}
    assert all(w["env_kwargs"]["toy_hand_off"] == "off" for w in up["worlds"])


def test_a_launch_without_the_setting_leaves_the_saved_value():
    state = _episode(finger_armature=0.02)
    up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(toy_hand_off="off"),
                                        env_settings="launch")
    assert changes == [] and up["worlds"][0]["env_kwargs"]["finger_armature"] == 0.02


def test_the_finger_armature_migrates_too():
    up, changes = CS.migrate_env_kwargs(_episode(), dict(finger_armature=0.02),
                                        env_settings="launch")
    assert {c["name"] for c in changes} == {"finger_armature"}
    assert all(w["env_kwargs"]["finger_armature"] == 0.02 for w in up["worlds"])


@pytest.mark.parametrize("bad", [dict(toy_hand_off="sometimes"), dict(finger_armature=-1.0),
                                 dict(finger_armature=float("nan"))])
def test_an_invalid_value_is_refused_before_anything_changes(bad):
    with pytest.raises(ValueError):
        CS.migrate_env_kwargs(_episode(), bad)


def test_switching_the_hand_off_off_while_the_infant_holds_the_toy_lets_go():
    state = _episode(toy_hand_off="follow")
    held = dict(held=True, hold_steps=10, free_steps=0, side="right", local_pos=[0, 0, 0.04],
                local_rot=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], adult_hand=[0, 0, 1], blend=0,
                grips=3, releases=2)
    state["worlds"][0]["world"]["caregiver"]["grip"] = held
    up, _ = CS.migrate_env_kwargs(state, dict(toy_hand_off="off"), env_settings="launch")
    g = up["worlds"][0]["world"]["caregiver"]["grip"]
    from mimo_bridge.caregiver import Caregiver
    assert g["held"] is False and g["local_pos"] is None and g["releases"] == 3
    assert g["blend"] == Caregiver.HAND_BLEND_STEPS and g["grips"] == 3
    assert state["worlds"][0]["world"]["caregiver"]["grip"]["held"] is True, "saved state edited"


def test_other_changed_settings_are_named_as_not_applied(caplog):
    state = _episode()
    launch = dict(state["worlds"][0]["env_kwargs"], scene="reach", seed=999, tag="_new",
                  run_dir="new/scenes", toy_hand_off="follow")
    with caplog.at_level(logging.INFO, logger="mimo_bridge.continuation_state"):
        CS.migrate_env_kwargs(state, dict(toy_hand_off="follow"), launch_kwargs=launch,
                              env_settings="launch")
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1 and "NOT applied" in warned[0] and "scene" in warned[0]
    for per_segment in ("seed", "tag", "run_dir"):
        assert per_segment not in warned[0].split("):")[-1], per_segment
    assert any("toy_hand_off set to 'follow' on 2 of 2" in r.getMessage() for r in caplog.records)


def test_a_setting_the_saved_world_never_had_is_judged_against_the_env_default(caplog, monkeypatch):
    """A flag added after the episode began: the saved world ran on BridgeEnv's default, so a
    launch value that differs from the default is ignored and must be named."""
    monkeypatch.setattr(CS, "_bridge_env_defaults", lambda: {"new_flag": "off", "other": 1})
    state = _episode()
    with caplog.at_level(logging.WARNING, logger="mimo_bridge.continuation_state"):
        CS.migrate_env_kwargs(state, {}, launch_kwargs=dict(new_flag="on", other=1))
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1 and warned[0].endswith(": new_flag")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mimo_bridge.continuation_state"):
        CS.migrate_env_kwargs(state, {}, launch_kwargs=dict(new_flag="off", other=1))
    assert not caplog.records, "a launch value equal to the default the world ran on is not a change"


def test_the_trainer_migrates_before_it_rebuilds_the_worlds():
    """The call must sit between loading the companion and building the sources from it."""
    src = (Path(__file__).resolve().parents[1] / "train_mimo.py").read_text(encoding="utf-8")
    load = src.index("episode_resume = continuation_state.load(args.resume)")
    call = src.index("continuation_state.migrate_env_kwargs(")
    build = src.index("saved_worlds=episode_resume['worlds']")
    assert load < call < build
    assert re.search(r"migrate_env_kwargs\(\s*episode_resume,\s*args,\s*launch_kwargs=kw", src)


def test_a_changed_hand_off_reaches_the_rebuilt_crib_world(tmp_path):
    """The real path: a crib world saved with the hand-off off, migrated, rebuilt from the
    migrated settings exactly as train_mimo builds a resumed single world, and restored."""
    from mimo_bridge.datasource import SingleEnvSource
    kw = dict(scene="crib", vision=False, audio_real=True, mobile=False, age=24,
              caregiver_mode="responsive", caregiver_speech_dir=str(SPEECH),
              run_dir=str(tmp_path), toy_hand_off="off")
    first = SingleEnvSource(**kw)
    try:
        first.reset()
        episode = dict(version=CS.VERSION, worlds=first.save_full_state(), config={},
                       trainer=dict(step=1))
    finally:
        first.close()
    assert episode["worlds"][0]["env_kwargs"]["toy_hand_off"] == "off"
    up, changes = CS.migrate_env_kwargs(episode, SimpleNamespace(toy_hand_off="follow"),
                                        env_settings="launch")
    assert len(changes) == 1
    saved_worlds = up["worlds"]
    src = SingleEnvSource(**saved_worlds[0]["env_kwargs"])      # train_mimo's resume line
    try:
        src.restore_full_state(saved_worlds)                     # refuses a mismatched config
        assert src.env_kwargs["toy_hand_off"] == "follow"
        assert src.env.caregiver.toy_hand_off == "follow"
        assert src.env.caregiver.diagnostics(src.env)["object_follows_grip"] is True
        # and the next checkpoint saves the migrated setting, so the lineage keeps it
        assert src.save_full_state()[0]["env_kwargs"]["toy_hand_off"] == "follow"
    finally:
        src.close()


def test_an_archived_checkpoint_loads_its_companion_from_the_archive(tmp_path, monkeypatch):
    """Checked 2026-09-22 on D:/.../seg31_toy_and_push/checkpoints/ckpt_step_340526: the
    archived receipt names the C: companion the archiver removed. load now finds it through
    the archiver's row, mirrored on C:, and still proves it against the receipt's hash."""
    import json
    import shutil
    from mimo_bridge import storage_locations as SL
    ck = tmp_path / "C" / "run" / "checkpoints" / "ckpt_step_7"
    ck.mkdir(parents=True)
    state = dict(version=CS.VERSION, worlds=[dict(payload=dict(sim_time=1.5))], learner={},
                 trainer=dict(step=7), config={})
    receipt = CS.write_episode(ck, state, storage_root=str(tmp_path / "C" / "companions"))
    c_comp = Path(receipt["storage_path"])
    d_comp = tmp_path / "D" / "checkpoint_state" / "run" / c_comp.name
    d_ck = tmp_path / "D" / "run" / "checkpoints" / ck.name
    d_comp.parent.mkdir(parents=True)
    shutil.copy2(c_comp, d_comp)
    shutil.copytree(ck, d_ck)
    c_manifest = tmp_path / "C" / "storage_moves.jsonl"
    SL.append_row(c_manifest, dict(event="moved", step=7, source=str(ck), destination=str(d_ck),
                                   companion_source=str(c_comp), companion_destination=str(d_comp),
                                   companion_sha256=receipt["sha256"],
                                   files={"episode_state.json": SL.sha256_file(d_ck / "episode_state.json")}))
    c_comp.unlink()
    shutil.rmtree(ck)
    monkeypatch.setattr(SL, "C_MANIFEST", c_manifest)
    monkeypatch.setattr(SL, "D_MANIFESTS", ())
    assert CS.load(d_ck)["trainer"]["step"] == 7            # the archive path
    assert CS.load(ck)["trainer"]["step"] == 7              # the old C: path of the checkpoint
    d_comp.write_bytes(b"not the companion")
    with pytest.raises(ValueError, match="integrity"):
        CS.load(d_ck)
    # an unknown companion is still the old error, not a silent None
    c_manifest.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        CS.load(d_ck)


def test_by_default_a_resume_keeps_every_saved_setting_and_names_the_ones_it_did_not_apply(caplog):
    """DESIGN D2: segment 31's command carries --toy-hand-off follow while its 8 worlds ran with the
    hand-off off (measured 2026-09-22). A resume of that command must change no world."""
    state = _episode(toy_hand_off=None, finger_armature=None)     # segment 31's companion shape
    before = copy.deepcopy(state)
    with caplog.at_level(logging.INFO, logger="mimo_bridge.continuation_state"):
        up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(toy_hand_off="follow",
                                                                   finger_armature=0.0004))
    assert up is state and changes == [] and state == before
    assert all("toy_hand_off" not in w["env_kwargs"] for w in up["worlds"])
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    named = [m for m in warned if m.startswith("Resume: env setting toy_hand_off")]
    assert len(named) == 1 and "NOT applied" in named[0] and "2 of 2" in named[0]
    assert "--resume-env-settings launch" in named[0]
    assert any(m.startswith("Resume: env setting finger_armature") for m in warned)
    assert not any("set to" in r.getMessage() for r in caplog.records)


def test_by_default_a_held_toy_is_not_released():
    """The grip release belongs to switching the hand-off off; with the saved settings kept,
    the saved grip stays exactly as it was."""
    state = _episode(toy_hand_off="follow")
    state["worlds"][0]["world"]["caregiver"]["grip"] = dict(held=True, hold_steps=4, grips=1,
                                                            releases=0)
    up, _ = CS.migrate_env_kwargs(state, dict(toy_hand_off="off"))
    assert up is state and up["worlds"][0]["world"]["caregiver"]["grip"]["held"] is True


def test_an_unknown_env_settings_value_is_refused():
    with pytest.raises(ValueError, match="env_settings"):
        CS.migrate_env_kwargs(_episode(), {}, env_settings="both")


def test_the_trainer_keeps_the_saved_settings_unless_launched_with_the_flag():
    """The call in train_mimo passes nothing (the default 'saved') or the --resume-env-settings
    flag, and that flag's default is 'saved'; either way a plain resume changes no world."""
    src = (Path(__file__).resolve().parents[1] / "train_mimo.py").read_text(encoding="utf-8")
    start = src.index("continuation_state.migrate_env_kwargs(")
    call = src[start:src.index(")", src.index("launch_kwargs=kw", start)) + 1]
    if "env_settings" in call:
        assert re.search(r"env_settings\s*=\s*args\.resume_env_settings", call), call
        flag = re.search(r"--resume-env-settings['\"].{0,120}?default=['\"](\w+)['\"]", src, re.S)
        assert flag and flag.group(1) == "saved", "the flag's default must be saved"
    else:
        import inspect
        assert inspect.signature(CS.migrate_env_kwargs).parameters["env_settings"].default == "saved"


def _crib_kwargs(run_dir, **kw):
    return dict(dict(scene="crib", vision=False, audio_real=True, mobile=False, age=24,
                     caregiver_mode="responsive", caregiver_speech_dir=str(SPEECH),
                     run_dir=str(run_dir), toy_hand_off="off"), **kw)


def test_a_two_body_pool_restores_migrated_worlds_and_keeps_saved_ones_by_default(tmp_path):
    """The 8-body production path, at 2 bodies: spawned workers each compare the saved env
    settings with the ones they were built with (datasource._worker_main) and refuse a mismatch.
    Built exactly as train_mimo builds a resumed pool (PoolSource(n, {}, per_env_kwargs=...)),
    once from worlds migrated with 'launch' and once from the default 'saved' (review ops-storage
    finding 6: the earlier test rebuilt the keyword merge inside the test and could not fail)."""
    from mimo_bridge.datasource import PoolSource
    first = PoolSource(2, _crib_kwargs(tmp_path))
    try:
        first.reset()
        episode = dict(version=CS.VERSION, worlds=first.save_full_state(), config={},
                       trainer=dict(step=1))
    finally:
        first.close()
    assert [w["env_kwargs"]["toy_hand_off"] for w in episode["worlds"]] == ["off", "off"]
    assert [w["env_kwargs"]["tag"] for w in episode["worlds"]] == ["_w0", "_w1"]
    for mode, want in (("launch", "follow"), ("saved", "off")):
        up, changes = CS.migrate_env_kwargs(episode, SimpleNamespace(toy_hand_off="follow"),
                                            env_settings=mode)
        assert len(changes) == (2 if mode == "launch" else 0)
        saved_worlds = up["worlds"]
        pool = PoolSource(2, {}, per_env_kwargs=[w["env_kwargs"] for w in saved_worlds])
        try:
            frames = pool.restore_full_state(saved_worlds)      # a mismatch raises in the worker
            assert len(frames) == 2
            again = pool.save_full_state()                      # the settings each worker runs
            assert [w["env_kwargs"]["toy_hand_off"] for w in again] == [want, want], mode
            assert [w["env_kwargs"]["seed"] for w in again] == \
                [w["env_kwargs"]["seed"] for w in episode["worlds"]], "a body's stream changed"
        finally:
            pool.close()
    # the control: a pool whose worker settings differ from the saved worlds is refused, so the
    # restore above is a real check and not a pass-through
    mismatched = [dict(w["env_kwargs"], toy_hand_off="follow") for w in episode["worlds"]]
    pool = PoolSource(2, {}, per_env_kwargs=mismatched)
    try:
        with pytest.raises(RuntimeError, match="mismatch"):
            pool.restore_full_state(episode["worlds"])
    finally:
        pool.close()
