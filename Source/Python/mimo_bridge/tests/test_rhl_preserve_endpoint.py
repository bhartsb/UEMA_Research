# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""preserve_endpoint.py (docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/): --step latest preserves the newest
COMPLETE checkpoint a run reached, so the completion job covers an early stop (the operator, Oct 11, 2026: "make the
backup job cover an early stop, using the actual checkpoint reached"); fake run folders, no model."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = (Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4" / "remembered_hand_learning_2026-10-11"
        / "preserve_endpoint.py")
spec = importlib.util.spec_from_file_location("preserve_endpoint", TOOL)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def fake_run(tmp, steps, incomplete=()):
    run = tmp / "run_x"
    comp_dir = tmp / "companions"
    comp_dir.mkdir(parents=True, exist_ok=True)
    for n in steps:
        ck = run / "checkpoints" / ("ckpt_step_%d" % n)
        ck.mkdir(parents=True)
        companion = comp_dir / ("ckpt_%d.pt.gz" % n)
        companion.write_bytes(b"worlds %d" % n)
        sha = hashlib.sha256(companion.read_bytes()).hexdigest()
        if n not in incomplete:
            (ck / "model.safetensors").write_bytes(b"weights %d" % n)
        (ck / "episode_state.json").write_text(json.dumps(dict(storage_path=str(companion), sha256=sha,
                                                               bytes=companion.stat().st_size)), encoding="utf-8")
    (run / "stage_done.json").write_text("{}", encoding="utf-8")
    return run


def test_latest_is_the_newest_complete_checkpoint(tmp_path):
    run = fake_run(tmp_path, [485000, 486000, 487000], incomplete=(487000,))
    assert P.newest_checkpoint_step(run) == 486000


def test_a_run_with_no_complete_checkpoint_is_refused(tmp_path):
    run = fake_run(tmp_path, [485000], incomplete=(485000,))
    with pytest.raises(SystemExit):
        P.newest_checkpoint_step(run)


def test_an_early_stop_is_preserved_at_the_checkpoint_it_reached(tmp_path):
    run = fake_run(tmp_path, [485000, 486731])
    latent = tmp_path / "latent"
    latent.mkdir()
    (latent / "latest.pt").write_bytes(b"latent")
    dest = tmp_path / "dest"
    assert P.main(["--run", str(run), "--step", "latest", "--latent", str(latent), "--dest", str(dest)]) == 0
    m = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
    assert m["step"] == 486731 and m["step_requested"] == "latest" and m["verdict"] == "pass"
    assert m["checkpoint"].endswith("ckpt_step_486731") and m["companion_equal"]
    assert any(f["path"] == "latent_body/latest.pt" for f in m["files"])


def test_an_existing_destination_is_refused(tmp_path):
    run = fake_run(tmp_path, [485000])
    dest = tmp_path / "dest"
    dest.mkdir()
    assert P.main(["--run", str(run), "--step", "latest", "--dest", str(dest)]) == 2
