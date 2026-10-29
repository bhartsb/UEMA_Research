# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.latent_body_runtime import example_record, jsonable
from mimo_bridge.tests.latent_body_fixtures import config, example, tree
from mimo_bridge.tools.latent_body import fit, load_learner, physical_measurements, main


def fit_arguments(tmp_path, overlap=False):
    (tmp_path / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    for name, body in (("data", 0), ("holdout", 0 if overlap else 1)):
        ex = example()
        for observation in (ex.history, ex.start, ex.end): observation.env_id = body
        (tmp_path / (name + ".jsonl")).write_text(json.dumps(example_record(ex), default=jsonable) + "\n", encoding="utf-8")
    return SimpleNamespace(config=str(tmp_path / "config.json"), data=str(tmp_path / "data.jsonl"),
        holdout=str(tmp_path / "holdout.jsonl"), output=str(tmp_path / "fit"), resume="",
        layout="trunk_v1", levels="arm,hand,index", updates=1, batch_size=1)


def test_offline_fit_preserves_checkpoint_and_reports_hindsight_separately(tmp_path):
    args = fit_arguments(tmp_path)
    receipt = fit(args)
    learner = load_learner(tmp_path / "fit" / "hierarchy.pt")
    assert learner.updates == 1 and learner.tree.fingerprint == tree().fingerprint
    assert receipt["completed"] and not receipt["physical_qualification"]
    assert receipt["held_out_body_episodes"] == [(1, 0)]
    with pytest.raises(ValueError, match="preserve"):
        fit(args)


def test_overlapping_holdout_refused_before_any_learning_and_cli_records_failure(tmp_path):
    args = fit_arguments(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        main(["fit", "--config", args.config, "--data", args.data, "--holdout", args.holdout,
              "--output", args.output, "--updates", "1", "--batch-size", "1"])
    assert not (tmp_path / "fit" / "hierarchy.pt").exists()
    assert json.loads((tmp_path / "fit" / "failure.json").read_text(encoding="utf-8"))["completed"] is False


def test_endpoint_following_and_wellness_use_declared_physical_readings():
    command = np.zeros((1, 2, 9)); command[..., 6] = 1
    outcome = dict(pos_cm=np.array([[0., 0., 0.], [1., 0., 0.]]),
                   quat=np.array([[0., 0., 0., 1.]] * 2), task={"pain": .25})
    measured = physical_measurements(command, outcome, start=np.ones((2, 3)),
        target=np.zeros((2, 3)), selected=np.array([0, 1]),
        lines=dict(following_cm=.1, following_rad=.1),
        wellness_fields={"pain": dict(path=["task", "pain"], units="synthetic pain units",
                                      measurement="synthetic fixture only")})
    assert measured["following_rate"] == .5 and measured["wellness"]["pain"]["value"] == .25
    assert measured["goal_improvement_cm2"] == pytest.approx(1 - 1/6)
    assert "not unreachable" in measured["interpretation"]


def test_shuffled_passes_keep_every_example_once_per_pass(tmp_path):
    from mimo_bridge.tools.latent_body import training_examples
    rows = []
    for step in range(1, 41):
        ex = example(step)
        rows.append(json.dumps(example_record(ex), default=jsonable))
    (tmp_path / "d.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    args = SimpleNamespace(data=str(tmp_path / "d.jsonl"), epochs=2, shuffle_buffer=8)
    frames = [ex.start.frame for ex in training_examples(args, 7)]
    assert sorted(frames) == sorted(list(range(1, 41)) * 2)
    assert frames[:40] != list(range(1, 41))
    plain = SimpleNamespace(data=str(tmp_path / "d.jsonl"))
    assert [ex.start.frame for ex in training_examples(plain, 7)] == list(range(1, 41))


def test_split_keeps_the_held_out_body_apart_and_scales_make_stillness_score_one(tmp_path):
    import torch
    from mimo_bridge.tools import latent_body_split
    from lem.latent_body.trajectory import make_batch
    rows = []
    for body in (0, 1):
        for step in range(1, 30):
            ex = example(step)
            ex.action_contract = "wrist_relative_joint_v1"
            for o in (ex.history, ex.start, ex.end):
                o.env_id = body
                o.bones[:, 4:7] += torch.randn(26, 3, generator=torch.Generator().manual_seed(step * 3 + body)) * (o.frame + 1)
            rows.append(dict(example_record(ex), ledger={"x": 1}))
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r, default=jsonable) + "\n" for r in rows), encoding="utf-8")
    base = dict(config(), action_contract="wrist_relative_joint_v1")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    latent_body_split.main(["--trajectory", str(tmp_path / "t.jsonl"), "--holdout-body", "1", "--base-config",
                            str(tmp_path / "base.json"), "--output", str(tmp_path / "s"), "--scale-rows", "29"])
    held = [json.loads(l) for l in (tmp_path / "s" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [json.loads(l) for l in (tmp_path / "s" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["start"]["env_id"] for r in held} == {1} and {r["start"]["env_id"] for r in train} == {0}
    assert "ledger" not in train[0]
    cfg = json.loads((tmp_path / "s" / "config.json").read_text(encoding="utf-8"))
    from mimo_bridge.tools.latent_body import read_examples
    b = make_batch(tree(), list(read_examples(tmp_path / "s" / "train.jsonl")), scales=cfg["scales"], duration_s=.05,
                   action_contract="wrist_relative_joint_v1", hand_frame="actual_wrist")
    for key in ("change", "action"):
        pooled = torch.cat([v.reshape(-1, 6 if key == "change" else 8) for v in getattr(b, key).values()])
        assert torch.allclose(pooled.square().mean(0)[pooled.square().mean(0) > 0], torch.tensor(1.), atol=1e-3)
