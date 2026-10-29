# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The shadow survives a supervisor relaunch with the same command, and its records stay under their cap."""
import json

from mimo_bridge.latent_body_runtime import ShadowRuntime
from mimo_bridge.tests.latent_body_fixtures import tree
from mimo_bridge.tests.test_latent_shadow_runtime import arguments, frame, transition


def test_auto_resume_continues_the_directorys_own_experience(tmp_path):
    args = arguments(tmp_path)
    args.latent_body_resume = "auto"
    first = ShadowRuntime(args, tree().handle_names)
    for step in range(3):
        first.observe([frame(step)], [] if step == 0 else [transition(step)], step, 0, train=True)
    assert first.learner.updates == 1 and first.latest.is_file()
    again = ShadowRuntime(args, tree().handle_names)          # the relaunch: same arguments
    again.observe([frame(3)], [transition(3)], 3, 0, train=True)
    assert again.learner.updates == 2


def test_records_rotate_at_their_cap(tmp_path):
    args = arguments(tmp_path)
    args.latent_body_log_mb = 0.001                            # 1,048 bytes
    runtime = ShadowRuntime(args, tree().handle_names)
    for step in range(12):
        runtime.observe([frame(step)], [] if step == 0 else [transition(step)], step, 0, train=True)
    main = runtime.directory / "trajectory.jsonl"
    rotated = runtime.directory / "trajectory.jsonl.1"
    assert rotated.is_file()
    for path in (main, rotated):
        rows = path.read_text(encoding="utf-8").splitlines()
        assert all(json.loads(r) for r in rows)
    assert main.stat().st_size < 2 * 1048 + max(len(r) for r in rotated.read_text(encoding="utf-8").splitlines()) + 2
