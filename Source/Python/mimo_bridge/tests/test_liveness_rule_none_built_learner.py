# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Review test (2026-09-26): --liveness-rule none must reach the learner the trainer builds (train_mimo.build_learner),
and every liveness repair parameter must be in the optimizer there. The branch's own test checks parse_args and
LemLite's signature only; a build_learner that drops the rule would revive units while args.json says none."""
import torch


def _built(tmp_path, rule):
    from mimo_bridge import train_mimo as TM
    argv = ["--out", str(tmp_path), "--liveness-repair", "on", "--no-vision", "--device", "cpu", "--n-envs", "1"]
    if rule is not None:
        argv += ["--liveness-rule", rule]
    args = TM.parse_args(argv)
    return TM.build_learner(args, None)


def test_the_rule_none_reaches_the_built_learner_and_its_repair_parts_train(tmp_path):
    m = _built(tmp_path / "none", "none")
    assert m.liveness is not None and m.liveness.mode == "on" and m.liveness.revive_rule == "none"
    assert m.receipt["liveness_repair"]["revive_rule"] == "none"
    in_opt = {id(p) for g in m.optimizer.param_groups for p in g["params"]}
    lv = list(m.liveness.parameters())
    assert lv and all(id(p) in in_opt for p in lv), "a liveness repair parameter is not trained"


def test_the_default_build_is_the_old_rule(tmp_path):
    m = _built(tmp_path / "default", None)
    assert m.liveness.revive_rule == "live_line" and "revive_rule" not in m.receipt["liveness_repair"]
