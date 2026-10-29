# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The existing training supervisor owns one durable comparison scheduler."""
from pathlib import Path
from types import SimpleNamespace

from mimo_bridge.tools import training_continuation as tc


def layout(tmp_path):
    cwd=tmp_path/'Source/Python';cwd.mkdir(parents=True)
    manifest=tmp_path/'docs/gain_of_function_v3/ability_rechecks.json'
    manifest.parent.mkdir(parents=True);manifest.write_text('{}')
    root=tmp_path/'Saved/mimo_bridge';root.mkdir(parents=True)
    return cwd,root


def test_matching_worker_prevents_another_launch(tmp_path,monkeypatch):
    cwd,root=layout(tmp_path)
    process=SimpleNamespace(pid=42,info=dict(cmdline=['python','-m','mimo_bridge.tools.ability_rechecks','--root',str(tmp_path)]))
    monkeypatch.setattr(tc.psutil,'process_iter',lambda fields:[process])
    monkeypatch.setattr(tc,'launch',lambda *args: (_ for _ in ()).throw(AssertionError('duplicate')))
    assert tc.ensure_ability_rechecks(root,cwd)['pids']==[42]


def test_start_passes_exact_workspace_and_manifest_then_backs_off(tmp_path,monkeypatch):
    cwd,root=layout(tmp_path);sent=[]
    monkeypatch.setattr(tc.psutil,'process_iter',lambda fields:[])
    monkeypatch.setattr(tc,'launch',lambda command,*args:sent.append(command) or SimpleNamespace(pid=42))
    assert tc.ensure_ability_rechecks(root,cwd,now=1000)['started_at']==1000
    assert Path(tc.argument(sent[0],'--root'))==tmp_path
    assert Path(tc.argument(sent[0],'--manifest')).is_file()
    assert 'delayed' in tc.ensure_ability_rechecks(root,cwd,last_start=1000,now=1100)['status']
    assert len(sent)==1


def test_unreadable_process_with_held_lock_does_not_duplicate(tmp_path,monkeypatch):
    cwd,root=layout(tmp_path)
    folder=root/'ability_rechecks';folder.mkdir()
    lock=tc.OwnerLock(folder/'owner.lock')
    monkeypatch.setattr(tc.psutil,'process_iter',lambda fields:[])
    monkeypatch.setattr(tc,'launch',lambda *args: (_ for _ in ()).throw(AssertionError('duplicate')))
    try:assert 'holds its lock' in tc.ensure_ability_rechecks(root,cwd)['status']
    finally:lock.close()
