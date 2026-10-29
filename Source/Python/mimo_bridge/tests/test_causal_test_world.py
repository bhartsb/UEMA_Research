# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The causal test runs in the run's own world and compares every body row, in both body layouts.

Found 2026-09-23 (fix-rows-r1-causal_test.py). run() built its world from ten hand-copied keys of
args.json; body_layout was not one of them, so a trunk_v1 checkpoint (every H84 segment: 26 body
rows, the lower trunk and the chest after the eyes' row 23) met a legacy 24-row world, and its first
26-row order stopped the test. Its check that a substitute order differs from the real one read rows
0-22 only, so a substitute that differed only in the lower trunk or the chest was thrown away as
identical. The first tests stand in for the model and the world; the last one builds a real world.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import causal_test as CT

LIMITS = {"phase_name": "mature", "rom_limit": 1.0, "strength_limit": 1.0, "age_months": 24.0,
          "exploration_noise": 0.0}
N = 10


def _run_args(layout, tmp_path):
    """The namespace _build_model returns, with segment 31's world settings (H84 args.json)."""
    from mimo_bridge.train_mimo import parse_args
    a = vars(parse_args(["--out", str(tmp_path / "unused_out")]))
    a.update(body_layout=layout, contact_profile="mechanical_v4", body_function="modulate",
             audio="real", ring_fov=150.0, critters="on", mobile="on", caregiver_mode="responsive",
             caregiver_seconds=15.0, caregiver_record_dir=str(tmp_path / "tape"),
             age_binding="fixed:24", delay_binding="off", scene_binding="phase",
             action_repeat=5, motor_hold=5, decoder_kp=6.0, seed=0, tag="")
    return a


def _model(layout):
    """Enough of LemLite for run(): the motor encoder's width, one real predicted sense and the
    eyes' row found by name, as LemLite finds it."""
    return SimpleNamespace(
        encoder_registry=SimpleNamespace(
            encoders={"motor": SimpleNamespace(patch_mlp=[torch.nn.Linear(19, 2)])}),
        ctx_len=4, active=["proprioception"], pred_mods=["proprioception"],
        config={"modalities": {"proprioception": {"data_source": "real"}}},
        GAZE_IDX=list(names_for(layout)).index("gaze"))


def _orders(layout, moved):
    """N transitions whose orders differ from one another ONLY in the named rows: 10 cm more
    forward per transition in each of them."""
    names = list(names_for(layout))
    out = []
    for k in range(N):
        act = torch.zeros(1, len(names), 9)
        act[..., 6] = 1.0
        act[..., 8] = 0.5
        for name in moved:
            act[0, names.index(name), 0] = 0.1 * k
        out.append({"t": {"proprioception": torch.zeros(1, 4)},
                    "t1": {"proprioception": torch.ones(1, 4)}, "act": act})
    return out


def _run(layout, moved, tmp_path, monkeypatch):
    """run() on stand-ins. The stand-in world counts its rows from the body_layout it is handed
    (BridgeEnv's default is legacy) and refuses an order with another row count, as the real
    SingleEnvSource.step does at its reshape."""
    import mimo_bridge.datasource as DS
    made = []

    class World:
        def __init__(self, **kw):
            made.append(kw)
            self.n_handles = len(names_for(kw.get("body_layout", "legacy")))

        def close(self):
            pass

    def collect(model, src, n, hold, limp=False, limits=None, babble=0.0):
        out = _orders(layout, moved)
        if out[0]["act"].shape[-2] != src.n_handles:
            raise ValueError(f"cannot reshape a {out[0]['act'].shape[-2]}-row order into "
                             f"{src.n_handles} rows")
        return out

    a = _run_args(layout, tmp_path)
    model = _model(layout)
    monkeypatch.setattr(CT, "_build_model", lambda ckpt, device, seed, intention=False:
                        (model, a, "args.json", True))
    monkeypatch.setattr(CT, "_limits_for_run", lambda ckpt, m: (dict(LIMITS), "test"))
    monkeypatch.setattr(CT, "_collect", collect)
    monkeypatch.setattr(CT, "_predict",
                        lambda m, tr, action, device: {"proprioception": torch.zeros(1, 4)})
    monkeypatch.setattr(DS, "SingleEnvSource", World)
    rep = CT.run("ckpt", N, 1, "crib", str(tmp_path / "scenes"), "cpu")
    return rep, made


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_the_probe_world_is_the_runs_world_with_the_probes_own_switches(layout, tmp_path, monkeypatch):
    rep, made = _run(layout, ("wrist_R",), tmp_path, monkeypatch)
    kw = made[0]
    # the run's body: before 2026-09-23 none of these reached the world
    assert kw["body_layout"] == layout
    assert (kw["contact_profile"], kw["body_function"], kw["audio_real"], kw["ring_fov"]) == \
        ("mechanical_v4", "modulate", True, 150.0)
    # the probe's own settings, as before
    assert (kw["scene"], kw["run_dir"], kw["vision"]) == ("crib", str(tmp_path / "scenes"), True)
    assert (kw["cmd_horizon"], kw["age"], kw["kp"], kw["caregiver_seconds"]) == (25, 24.0, 6.0, 15.0)
    assert (kw["critters"], kw["mobile"], kw["caregiver_mode"]) == (False, False, "off")
    # never records to or replays the run's caregiver tape
    assert (kw["caregiver_record_dir"], kw["caregiver_replay"]) == ("", "")
    # each stream its own seed and tag: this body, then a different body
    assert [(k["tag"], k["seed"]) for k in made] == [("_act0", 100), ("_oth0", 300)]
    # the report records the world it ran in
    assert rep["config"]["env_kw"]["body_layout"] == layout


@pytest.mark.parametrize("layout,moved,skipped", [
    ("trunk_v1", ("lower_trunk", "chest"), 0),   # before 2026-09-23: all 10 thrown away as identical
    ("trunk_v1", ("gaze",), N),                  # the eyes' row alone is no body order (review E14)
    ("legacy", ("wrist_R",), 0),
    ("legacy", ("gaze",), N)])
def test_a_substitute_that_differs_in_any_body_row_but_the_eyes_is_used(layout, moved, skipped,
                                                                          tmp_path, monkeypatch):
    rep, _ = _run(layout, moved, tmp_path, monkeypatch)
    assert rep["per_seed"][0]["skipped_identical"] == {"shuffled": skipped, "cross-body": skipped}
    used = rep["per_seed"][0]["mse"]["shuffled"]["proprioception"]
    assert (used == used) == (skipped == 0)      # scored only when some substitute was used


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_the_probe_world_takes_an_order_with_every_checkpoint_row(layout, tmp_path):
    """A real world built from the probe's keywords: one row per row of the checkpoint's layout,
    and a whole order of that size steps. Vision off (no rows depend on it)."""
    from mimo_bridge.datasource import SingleEnvSource
    kw = CT.world_kwargs(_run_args(layout, tmp_path), dict(LIMITS), "crib", str(tmp_path / "scenes"), 25)
    kw["vision"] = False
    src = SingleEnvSource(tag="_ctworld", seed=100, **kw)
    try:
        n = len(names_for(layout))
        assert src.n_handles == n
        src.reset()
        order = np.zeros((1, n, 9), np.float32)
        order[..., 6] = 1.0
        src.step(order, np.zeros((1, 2), np.float32), None, render=False)
    finally:
        src.close()
    assert not (tmp_path / "tape").exists()
