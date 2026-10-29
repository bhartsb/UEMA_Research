# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The restart must retain its source and stop even with a queued snapshot."""
import importlib
import hashlib
import json
from pathlib import Path

import pytest
from types import SimpleNamespace


def modules(monkeypatch):
    folder = Path(__file__).resolve().parents[4] / 'docs/gain_of_function_v3'
    monkeypatch.syspath_prepend(str(folder))
    return (importlib.import_module('prove_planner_repair'),
            importlib.import_module('resume_after_planner_repair'))


def test_restart_failure_replaces_pending_snapshot_and_pauses_supervisor(tmp_path, monkeypatch):
    _, resume = modules(monkeypatch)
    run = tmp_path / 'run'
    run.mkdir()
    pause = tmp_path / 'supervisor_control.json'
    pause.write_text(json.dumps({'cmd': 'run'}))
    (run / 'control.json').write_text(json.dumps({'cmd': 'snapshot'}))
    monkeypatch.setattr(resume, 'OUT', run)
    monkeypatch.setattr(resume.D, 'PAUSE', pause)
    with pytest.raises(RuntimeError, match='supervisor remains paused'):
        resume.fail('Injected startup failure')
    assert json.loads(pause.read_text())['cmd'] == 'pause'
    assert json.loads((run / 'control.json').read_text())['cmd'] == 'snapshot_and_stop'
    assert json.loads((run / 'stop_reason.json').read_text())['resume_policy'] == 'repair_required'


@pytest.mark.parametrize('changed', ['model.safetensors', 'optimizer.pt', 'runtime_state.pt'])
def test_private_source_must_match_original_learning_state(tmp_path, monkeypatch, changed):
    proof, _ = modules(monkeypatch)
    original, private = tmp_path / 'original', tmp_path / 'private'
    for folder in (original, private):
        folder.mkdir()
        for name in proof.SOURCE_FILES:
            (folder / name).write_bytes(name.encode())
        (folder / 'episode_state.json').write_text(json.dumps({'step': 372375, 'times': list(range(8))}))
    assert proof.verify_source_copy(original, private) == proof.source_hashes(original)
    (private / changed).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='differs from original'):
        proof.verify_source_copy(original, private)


def test_private_source_must_keep_all_eight_world_times(tmp_path, monkeypatch):
    proof, _ = modules(monkeypatch)
    original, private = tmp_path / 'original', tmp_path / 'private'
    for folder in (original, private):
        folder.mkdir()
        for name in proof.SOURCE_FILES:
            (folder / name).write_bytes(name.encode())
        (folder / 'episode_state.json').write_text(json.dumps({'step': 372375, 'times': list(range(8))}))
    (private / 'episode_state.json').write_text(json.dumps({'step': 372375, 'times': [0] * 8}))
    with pytest.raises(AssertionError):
        proof.verify_source_copy(original, private)


def test_episode_receipt_accepts_external_file_and_rejects_corruption(tmp_path, monkeypatch):
    proof,_=modules(monkeypatch)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    external=tmp_path/'physical_history.pt.gz';external.write_bytes(b'complete saved episode')
    receipt=dict(version='episode_continuation_v1',storage_path=str(external),
                 sha256=hashlib.sha256(external.read_bytes()).hexdigest(),bytes=external.stat().st_size,
                 step=372375,times=list(range(8)))
    (checkpoint/'episode_state.json').write_text(json.dumps(receipt))
    assert proof.episode_integrity(checkpoint)['path']==str(external)
    external.write_bytes(b'partial')
    with pytest.raises(RuntimeError,match='receipt'):
        proof.episode_integrity(checkpoint)


def test_windows_supervisor_child_must_belong_to_the_launched_process(tmp_path,monkeypatch):
    _,resume=modules(monkeypatch)
    import psutil
    process=SimpleNamespace(pid=200,parents=lambda:[SimpleNamespace(pid=100)],
        cmdline=lambda:['python','-m','mimo_bridge.tools.training_continuation','--run',str(tmp_path)])
    monkeypatch.setattr(psutil,'Process',lambda pid:process)
    state=dict(supervisor_pid=200,active_run=str(tmp_path))
    assert resume.supervisor_owned_by_launcher(state,100,tmp_path)
    assert resume.supervisor_owned_by_launcher(state,200,tmp_path)
    assert not resume.supervisor_owned_by_launcher(state,999,tmp_path)
    assert not resume.supervisor_owned_by_launcher(state,100,tmp_path/'different_run')
