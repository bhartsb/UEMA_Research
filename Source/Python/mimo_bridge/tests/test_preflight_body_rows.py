# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The launch preflight tests the run's own body, ordering every row of it by name.

What was wrong (found 2026-09-23, docs/gain_of_function_v3/agents/fix-rows-r1-preflight.py.md):
tools/preflight.run_probes, the gate train_mimo runs before every segment and refuses the launch
on, built its world from its own keyword set, which never carried body_layout or contact_profile.
BridgeEnv's default body_layout is 'legacy', so every trunk_v1 run (26 rows: the lower trunk and
chest after the eyes' row; all 50 H84 segments) was certified on a 24-row body with no trunk
handles, and 49 H84 preflight.json files read PASS for a body the run never had. Its orders were
fixed tables of 24 rows and its limb was row 15 by position, so passing the layout alone would
have crashed the gate (row 24 of a 24-row table) and refused every trunk_v1 launch.

Now train_mimo passes the settings of the world it builds, the command line rebuilds them with
train_mimo.env_kwargs_for, every order has one row per body row, the arm is found by name, and a
trunk_v1 body also has its chest and lower trunk ordered and scored.
"""
import json
import os

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for


class _Stop(Exception):
    pass


def _stub_world(monkeypatch):
    """Replace BridgeEnv with a recorder that stops the gate before any physics."""
    import mimo_bridge.env as E
    seen = {}

    def stub(**kw):
        seen.update(kw)
        raise _Stop()
    monkeypatch.setattr(E, "BridgeEnv", stub)
    return seen


def _args(tmp_path, **over):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    a = dict(scene="crib", scene_binding="phase", curriculum="off", decoder_kp=6.0,
             caregiver_record_dir=str(tmp_path / "lineage_records"), seed=0, tag="")
    a.update(over)
    p = run / "args.json"
    p.write_text(json.dumps(a), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_the_command_line_gate_builds_the_runs_own_body(tmp_path, monkeypatch, layout):
    from mimo_bridge.tools import preflight
    over = dict(body_layout=layout, contact_profile="mechanical_v4") if layout == "trunk_v1" else {}
    path = _args(tmp_path, **over)
    seen = _stub_world(monkeypatch)
    with pytest.raises(_Stop):
        preflight.run_probes(path)
    assert seen.get("body_layout") == layout, "the gate must build the layout args.json names"
    assert len(names_for(seen["body_layout"])) == (26 if layout == "trunk_v1" else 24)
    assert seen.get("contact_profile") == ("mechanical_v4" if layout == "trunk_v1" else "legacy")
    assert seen["kp"] == 6.0 and seen["scene"] == "crib"
    # a throwaway world: fixed seed, eyes on, scene files beside the run, never recording into
    # the lineage's shared caregiver event store
    assert seen["seed"] == 1234 and seen["tag"] == "_preflight" and seen["vision"] is True
    assert seen["run_dir"] == os.path.join(os.path.dirname(path), "scenes")
    assert seen["caregiver_record_dir"] == ""


def test_the_trainers_own_settings_reach_the_world_unchanged(tmp_path, monkeypatch):
    from mimo_bridge.tools import preflight
    path = _args(tmp_path, scene_binding="fixed")
    given = dict(scene="crib", body_layout="trunk_v1", contact_profile="mechanical_v4", age=24.0,
                 mobile=True, critters=True, kp=6.0, seed=7, tag="x", vision=False,
                 run_dir=str(tmp_path / "elsewhere"), caregiver_record_dir=str(tmp_path / "rec"))
    before = dict(given)
    seen = _stub_world(monkeypatch)
    with pytest.raises(_Stop):
        preflight.run_probes(path, env_kw=given)
    assert given == before, "the trainer's dict must not be edited by the gate"
    expected = dict(before, seed=1234, tag="_preflight", vision=True, caregiver_record_dir="",
                    run_dir=os.path.join(os.path.dirname(path), "scenes"))
    assert seen == expected


def _trainer_flags(tmp_path, layout):
    return ['--out', str(tmp_path / 'run'), '--device', 'cpu', '--n-envs', '1', '--no-vision',
            '--curriculum', 'off', '--age-binding', 'none', '--delay-binding', 'off', '--episode-steps', '0',
            '--action-repeat', '1', '--motor-hold', '2', '--minutes', '0', '--max-steps', '1', '--no-babble',
            '--self-state', 'perceive', '--body-function', 'modulate', '--persistent-memory', 'on',
            '--imagination', 'off', '--next-frame-loss', 'on', '--contact-profile', 'mechanical_v4',
            '--replay', 'off', '--report-every', '1', '--video-every-min', '0',
            '--body-layout', layout, '--decoder-kp', '6.0',
            '--checkpoint-every-min', '0', '--checkpoint-storage', str(tmp_path / 'storage' / 'checkpoints')]


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_the_trainer_hands_the_gate_the_world_it_builds(tmp_path, monkeypatch, layout):
    """Driven through the real trainer on CPU; the gate is stubbed to fail so the launch stops
    there, before any world is built."""
    import torch
    from mimo_bridge.tools import preflight
    from mimo_bridge.train_mimo import main
    got = {}

    def stub(path, env_kw=None):
        got.update(path=path, env_kw=env_kw)
        return dict(ok=False, checks=[dict(name="stub", ok=False)])
    monkeypatch.setattr(preflight, "run_probes", stub)
    torch.set_num_threads(1)
    with pytest.raises(SystemExit):
        main(_trainer_flags(tmp_path, layout))
    kw = got.get("env_kw")
    assert kw is not None, "the trainer must pass the settings of the world it builds"
    assert kw["body_layout"] == layout and kw["contact_profile"] == "mechanical_v4"
    assert kw["kp"] == 6.0 and kw["vision"] is False
    assert got["path"] == os.path.join(str(tmp_path / "run"), "args.json")


def _world_kw(layout):
    """A crib world like H84's body (24 months, mechanical_v4 contact, kp 6), no caregiver."""
    return dict(scene="crib", age=24.0, contact_profile="mechanical_v4", kp=6.0,
                caregiver_seconds=0.0, body_layout=layout)


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_the_gate_orders_every_row_of_the_body_it_tests(tmp_path, layout):
    from mimo_bridge.tools import preflight
    path = _args(tmp_path, scene_binding="fixed")
    rep = preflight.run_probes(path, env_kw=_world_kw(layout))
    names = {c["name"]: c for c in rep["checks"]}
    assert rep["body_layout"] == layout and rep["n_handles"] == len(names_for(layout))
    assert rep["contact_profile"] == "mechanical_v4"
    arm = names["limb follows a command over 250 ms: cosine"]
    assert arm["part"] == "upperarm_R" and arm["ok"], arm
    trunk = [c for c in rep["checks"] if c.get("part") in ("chest", "lower_trunk")]
    if layout == "trunk_v1":
        assert sorted(c["part"] for c in trunk) == ["chest", "lower_trunk"]
        assert all(c["ok"] for c in trunk), trunk
    else:
        assert trunk == [], "a legacy body has no trunk rows to order"
    assert rep["ok"], [c for c in rep["checks"] if not c["ok"]]


@pytest.mark.slow
@pytest.mark.parametrize("part", ["lower_trunk", "chest"])
def test_a_trunk_order_that_reaches_no_joint_fails_the_gate(tmp_path, monkeypatch, part):
    """The control for the trunk checks: cut one trunk handle off its joints and the gate must
    refuse, while the other trunk part still passes."""
    import mimo_bridge.env as E
    Real = E.BridgeEnv

    class Cut(Real):
        def __init__(self, **kw):
            super().__init__(**kw)
            h = self.decoder.handles[list(self.proprio.handle_names).index(part)]
            for lst in (h.joint_ids, h.dof_adrs, h.act_ids, h.axis_body_ids):
                lst.clear()
    monkeypatch.setattr(E, "BridgeEnv", Cut)
    from mimo_bridge.tools import preflight
    path = _args(tmp_path, scene_binding="fixed")
    rep = preflight.run_probes(path, env_kw=_world_kw("trunk_v1"))
    by_part = {c["part"]: c for c in rep["checks"] if c.get("part") in ("chest", "lower_trunk")}
    other = "chest" if part == "lower_trunk" else "lower_trunk"
    assert not by_part[part]["ok"], by_part[part]
    assert by_part[other]["ok"], by_part[other]
    assert not rep["ok"]
    assert np.isfinite(by_part[part]["value"])


@pytest.mark.slow
def test_wrist_relative_hand_contract_passes_real_physical_preflight(tmp_path):
    """Zero-strength eye checks and full-strength limb checks share this path."""
    from mimo_bridge.tools import preflight
    path=_args(tmp_path,scene_binding='fixed',hand_target_mode='wrist_relative_v1')
    settings=dict(_world_kw('trunk_v1'),hand_target_mode='wrist_relative_v1')
    report=preflight.run_probes(path,env_kw=settings)
    assert report['n_handles']==26
    assert report['ok'],[check for check in report['checks'] if not check['ok']]
