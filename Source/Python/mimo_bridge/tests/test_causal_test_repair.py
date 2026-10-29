# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""causal_test's model builder and prediction route, repaired 2026-09-22.

Three things are pinned here:
  - the builder finds the run's args.json beside the checkpoint, through lineage.json's
    "out" (archived checkpoints on D: have no args.json beside them), or reports it absent;
  - the namespace it builds from is the one self_recovery_experiment.configuration() builds
    (the trainer's parse_args defaults overwritten by the run's args.json), so the model is
    made by train_mimo.build_learner the way the trainer makes it;
  - _collect records the persistent-memory bias and the back-filled context the live
    next-frame lesson uses, and _predict hands that bias to the forward model.
The last test loads the real H84 checkpoint, found wherever the archiver moved it, and is skipped
only when it is nowhere.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.tools import causal_test as CT


def _run(tmp_path, name, args):
    run = tmp_path / name
    ckpt = run / "checkpoints" / "ckpt_step_7"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps(args), encoding="utf-8")
    return run, ckpt


def test_args_found_beside_the_checkpoint(tmp_path):
    run, ckpt = _run(tmp_path, "R", {"context_length": 8})
    path, found = CT._run_args_path(str(ckpt))
    assert found and os.path.samefile(path, run / "args.json")


def test_args_found_through_lineage_for_an_archived_checkpoint(tmp_path):
    run, _ = _run(tmp_path, "R", {"context_length": 8, "seed": 5})
    archived = tmp_path / "D" / "archive" / "R" / "checkpoints" / "ckpt_step_9"
    archived.mkdir(parents=True)
    (archived / "lineage.json").write_text(json.dumps({"out": str(run)}), encoding="utf-8")
    saved, path, found = CT._cfg_from_run(str(archived))
    assert found and saved["seed"] == 5
    assert os.path.samefile(path, run / "args.json")


def test_missing_args_is_reported_not_invented(tmp_path):
    lone = tmp_path / "X" / "checkpoints" / "ckpt_step_1"
    lone.mkdir(parents=True)
    saved, path, found = CT._cfg_from_run(str(lone))
    assert saved == {} and not found


def test_the_namespace_is_self_recovery_configuration(tmp_path):
    """The probe namespace equals self_recovery_experiment.configuration() except for the
    fields the caller owns (output folder, seed, intention and the planner-only flags)."""
    from pathlib import Path
    from mimo_bridge.tools.self_recovery_experiment import configuration
    saved = {"context_length": 16, "d_model": 192, "core_blocks": 4, "residual_prediction": "on",
             "self_prediction": "grounded_residual", "persistent_memory": "on",
             "self_choice": "off", "intention": "on", "agency_drive": "on", "seed": 3}
    run, ckpt = _run(tmp_path, "R", saved)
    reference = vars(configuration(Path(ckpt), tmp_path / "out"))
    ours = vars(CT._probe_configuration(saved, "cpu", 11, True, None))
    skip = {"out", "seed"}
    diff = {k for k in set(reference) | set(ours) if k not in skip and reference.get(k) != ours.get(k)}
    assert not diff, {k: (reference.get(k), ours.get(k)) for k in diff}
    assert ours["seed"] == 11
    off = vars(CT._probe_configuration(saved, "cpu", 0, False, None))
    assert off["intention"] == "off" and off["agency_drive"] == "off"
    # overrides are applied last and recorded by the caller
    over = vars(CT._probe_configuration(saved, "cpu", 0, True, {"context_length": 32}))
    assert over["context_length"] == 32


def test_configuration_reads_an_archived_checkpoints_args_through_its_lineage(tmp_path):
    """2026-09-26: configuration() read <checkpoint>/../../args.json only, which an archived
    checkpoint on D: does not have (the archive holds checkpoints only); the five checkpoint proofs
    that build through it could not load DESIGN D3's baseline from the archive."""
    from pathlib import Path
    from mimo_bridge.tools.self_recovery_experiment import configuration
    run, _ = _run(tmp_path, "R", {"context_length": 16, "seed": 5})
    archived = tmp_path / "D" / "archive" / "R" / "checkpoints" / "ckpt_step_9"
    archived.mkdir(parents=True)
    (archived / "lineage.json").write_text(json.dumps({"out": str(run)}), encoding="utf-8")
    a = configuration(archived, tmp_path / "out")
    assert a.context_length == 16 and a.seed == 5 and a.n_envs == 1
    # the control: without the lineage record the run's arguments are nowhere, and it says so
    (archived / "lineage.json").unlink()
    with pytest.raises(FileNotFoundError, match="no args.json"):
        configuration(Path(archived), tmp_path / "out")


def test_build_returns_the_runs_seed_and_full_namespace(tmp_path, monkeypatch):
    import mimo_bridge.train_mimo as TM
    run, ckpt = _run(tmp_path, "R", {"context_length": 4, "seed": 17, "motor_hold": 5})
    built = {}

    def fake_build(ns, ckpt_dir):
        built["ns"] = ns
        built["dir"] = ckpt_dir
        return SimpleNamespace(config={})

    monkeypatch.setattr(TM, "build_learner", fake_build)
    m, a, src, found = CT._build_model(str(ckpt), "cpu", 2, load=False)
    assert found and a["seed"] == 17 and built["ns"].seed == 2
    assert built["dir"] is None                 # no checkpoint folder created by a probe
    assert a["motor_hold"] == 5 and "action_repeat" in a     # trainer defaults filled in
    assert m.config["training"]["steps_per_phase"] > 0


class _Recorder:
    """A forward model that records what it was handed."""

    def __init__(self, d):
        self.d = d
        self.kw = None

    def __call__(self, tokens, ctx, **kw):
        self.kw = kw
        B = next(iter(tokens.values())).shape[0]
        return None, torch.zeros(B, len(tokens), self.d), None


def _fake_model(d=4):
    enc = SimpleNamespace(encode=lambda frame, mask: {"motor": torch.zeros(1, 1, d)})

    class Heads:
        head_modalities = ("vision",)

        def __call__(self, last, current=None, who=None):
            return dict(current or {})

    return SimpleNamespace(encoder_registry=enc, active=["vision", "motor"], d_model=d,
                           forward_model=_Recorder(d), prediction_heads=Heads())


def test_predict_passes_the_memory_bias_the_live_lesson_uses():
    model = _fake_model()
    tr = {"t": {"vision": torch.ones(1, 4)}, "ach": torch.zeros(1, 26, 10),
          "ctx": torch.zeros(1, 12, 4), "mem": torch.full((1, 6, 4), 0.5)}
    act = torch.zeros(1, 26, 9)
    act[..., 6] = 1.0
    CT._predict(model, tr, act, "cpu")
    assert torch.equal(model.forward_model.kw["memory_bias"], tr["mem"])
    # a record without memory predicts as before
    del tr["mem"]
    CT._predict(model, tr, act, "cpu")
    assert "memory_bias" not in model.forward_model.kw


class _StepModel:
    """Enough of LemLite for _collect: each step writes a pre-shift context marker into
    _nf_ctx and a memory bias, then pushes the buffer."""

    def __init__(self):
        self.ctx_len, self.n_mod, self.d_model = 3, 2, 4
        self.context_buffer = None
        self.step_n = 0

    def reset_transition_state(self):
        self.context_buffer = None
        self._nf_ctx = None

    def step(self, frames, dev, train, step_index, executed_action, executed_order_id):
        shape = (1, (self.ctx_len - 1) * self.n_mod, self.d_model)
        if self.context_buffer is None:
            self.context_buffer = torch.zeros(shape)
        # the lesson's context: this step's number (back-filled, pre-push)
        self._nf_ctx = torch.full(shape, float(self.step_n) + 0.5)
        self._memory_bias_current = torch.full((1, 5, 4), float(self.step_n))
        self.context_buffer = torch.full(shape, float(self.step_n) + 0.9)   # post-push
        self.last_tokens = {"vision": torch.full((1, 4), float(self.step_n))}
        self.last_achieved = torch.zeros(1, 26, 10)
        self.step_n += 1
        motor = np.zeros((1, 26, 9), np.float32)
        motor[..., 6] = 1.0
        return SimpleNamespace(motor_for_action=motor, gaze_out=np.zeros((1, 2), np.float32))


class _Src:
    env = None

    def reset(self):
        return [SimpleNamespace(meta={})]

    def step(self, motor, gaze, mask, render=False):
        return [SimpleNamespace(meta={"gaze_cmd": (0.0, 0.0), "cmd_response": {"cmd_mag": 0.1}})]


def test_collect_records_the_lesson_context_and_memory():
    out = CT._collect(_StepModel(), _Src(), 4, 5, limits={"_action_repeat": 2})
    assert len(out) == 3               # transitions close one step late
    for i, rec in enumerate(out):
        # the record for S_i carries S_i's lesson context and memory, not the post-push buffer
        assert float(rec["ctx"].mean()) == pytest.approx(i + 0.5)
        assert float(rec["mem"].mean()) == pytest.approx(i)
        assert float(rec["t1"]["vision"].mean()) == pytest.approx(i + 1)


# found wherever the archiver moved it (tests/saved_checkpoints.py); skipped only where it is nowhere
from mimo_bridge.tests.saved_checkpoints import h84_344400  # noqa: E402

H84_FOUND = h84_344400()
H84 = None if H84_FOUND.path is None else str(H84_FOUND.path)


@pytest.mark.slow
@H84_FOUND.mark()
def test_h84_loads_with_nothing_reinitialised():
    """Measured refusal before the repair: "checkpoint has learned grounded self dynamics;
    retain --self-prediction grounded_residual"."""
    H84_FOUND.require()
    m, a, src, found = CT._build_model(H84, "cpu", 0, load=False)
    result = m.load_checkpoint(H84, load_optimizer=False)
    assert found and a["self_prediction"] == "grounded_residual"
    assert result.get("reinitialized_components") == []
    assert m.d_model == 192 and m.persistent_memory is not None
