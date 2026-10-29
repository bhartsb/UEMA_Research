# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The explainer's two producers: the live trainer hook and the checkpoint probe.

The probe test pins gap-register finding 1 (a bare `train_mimo --resume --eval-only`
builds a ctx-4 model against a ctx-64 checkpoint); the trainer tests run train_mimo
for real - a handful of steps on one body, vision off - because the hook, the control
command and the writer only prove themselves in the loop they live in.
"""
import glob
import json
import os

import pytest

from mimo_bridge import train_mimo as TM
from mimo_bridge.explainer import schema as SCH
from mimo_bridge.tools.explain_probe import derive_argv


# ------------------------------------------------------------------ derive_argv
H62_SHAPED_ARGS = {
    "n_envs": 13, "minutes": 480.0, "max_steps": 0, "scene": "crib",
    "scene_binding": "fixed", "out": "Saved/mimo_bridge/experiments/H62_swept",
    "device": "cuda:0", "seed": 0, "render_every": 2, "no_vision": False,
    "placeholder_senses": "on", "residual_prediction": "on", "reinit": "",
    "fresh_optimizer": False, "fill_ring_first": True, "gaze_reflex": "on",
    "report_every": 100, "episode_steps": 2000, "curriculum": "on",
    "steps_per_phase": 3000, "start_phase": "child", "age_binding": "curriculum",
    "delay_binding": "on", "wellness_mode": "baseline",
    "motor_state_formula": "boundary", "motor_hold": 5, "imagination": "config",
    "random_actions": False, "checkpoint_every_min": 60.0, "eval_only": False,
    "no_babble": False, "babble_noise": 0.05, "babble_scale": 1.0, "limp": False,
    "replay": "on", "replay_every": 4, "replay_batch": 48, "replay_size": 6000,
    "think_harder": "off", "intention": "on", "intention_every": 5,
    "no_intention_execute": False, "action_tenure_loss": "on",
    "action_tenure_weight": 0.5, "action_tenure_token_reach": "on",
    "action_tenure_token_weight": 0.25, "agency_drive": "on",
    "movement_drive": "on", "next_frame_loss": "on", "top_down_gate": "on",
    "smooth_actions": 0.3, "context_length": 64, "fovea_size": 288,
    "fovea_fov": 24.0, "action_repeat": 5, "video_every_min": 20.0,
    "video_seconds": 30.0, "resume": "some/older/ckpt", "tag": "",
    "curiosity_band": "0.02:0.4", "probe_touch": False,
}


def test_derive_argv_replicates_the_run_and_forces_probe_safety(tmp_path):
    out = str(tmp_path / "probe_out")
    argv = derive_argv(H62_SHAPED_ARGS, {"phase": "mature"},
                       ckpt="X/checkpoints/ckpt_step_9", out=out,
                       n_steps=200, every=10, n_envs=2)
    a = TM.parse_args(argv)                     # must round-trip the real parser
    # model-shaping flags came from the RUN, not the CLI defaults
    assert a.context_length == 64
    assert a.residual_prediction == "on"
    assert a.next_frame_loss == "on"
    assert a.top_down_gate == "on"
    assert a.action_tenure_token_reach == "on"
    assert a.smooth_actions == 0.3
    assert a.fovea_size == 288 and a.fovea_fov == 24.0
    assert a.curiosity_band == "0.02:0.4"
    # probe safety: eval-only, fresh out, lineage phase, no videos, its own resume
    assert a.eval_only is True
    assert a.out == out
    assert a.start_phase == "mature"
    assert a.video_every_min == 0
    assert a.resume == "X/checkpoints/ckpt_step_9"
    assert a.max_steps == 200 and a.n_envs == "2"
    assert a.explain_steps == 10
    # run-specific keys were NOT inherited
    assert a.minutes == 60.0 and a.tag == "_explainprobe"


def test_derive_argv_falls_back_without_lineage():
    argv = derive_argv(H62_SHAPED_ARGS, None, ckpt="c", out="o")
    a = TM.parse_args(argv)
    assert a.start_phase == "child"             # the run's own start_phase
    argv2 = derive_argv(H62_SHAPED_ARGS, {"phase": "fixed"}, ckpt="c", out="o")
    assert TM.parse_args(argv2).start_phase == "child"


# ------------------------------------------------------------------ the live hook
def _run_trainer(tmp_path, extra, pre_control=None):
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    if pre_control:
        with open(os.path.join(out, "control.json"), "w") as fh:
            json.dump(pre_control, fh)
    # --device cpu: the trainer's default is cuda:0, and the graphics card belongs to the training
    # run. On cuda:0 both micro-runs stopped at their first model step on 2026-09-25 ("Invalid
    # device id" from torch's bf16 query); on the CPU the model runs in float32 with autocast off
    argv = ["--n-envs", "1", "--no-vision", "--minutes", "5", "--device", "cpu",
            "--scene", "reach", "--out", out, "--video-every-min", "0",
            "--replay", "off", "--curriculum", "on"] + extra
    rc = TM.main(argv)
    assert rc == 0, "trainer micro-run failed"
    return out


@pytest.mark.slow
def test_trainer_writes_snapshots_on_cadence(tmp_path):
    out = _run_trainer(tmp_path, ["--max-steps", "5", "--explain-steps", "2",
                                  "--next-frame-loss", "on"])
    assert "explain" in open(os.path.join(out, "control_supported")).read().split()
    files = sorted(glob.glob(os.path.join(out, "explain", "step_*.json")))
    assert [os.path.basename(f) for f in files] == \
        ["step_0.json", "step_2.json", "step_4.json"]
    snap = json.load(open(files[-1], encoding="utf-8"))
    ok, errs = SCH.validate(snap)
    assert ok, errs
    assert snap["header"]["weights"] == "post_update"
    assert snap["header"]["step"] == 4
    # of_now is real by step 4 with --next-frame-loss on
    assert isinstance(snap["predictions"]["of_now"], dict)
    idx = json.load(open(os.path.join(out, "explain", "index.json")))
    assert [e["step"] for e in idx["snapshots"]] == [0, 2, 4]


@pytest.mark.slow
def test_explain_control_command_takes_one_snapshot(tmp_path):
    out = _run_trainer(tmp_path, ["--max-steps", "3"],
                       pre_control={"cmd": "explain"})
    files = glob.glob(os.path.join(out, "explain", "step_*.json"))
    assert len(files) == 1, "the command is one-shot: exactly one snapshot"
    assert not os.path.exists(os.path.join(out, "control.json")), "command consumed"
    snap = json.load(open(files[0], encoding="utf-8"))
    ok, errs = SCH.validate(snap)
    assert ok, errs
