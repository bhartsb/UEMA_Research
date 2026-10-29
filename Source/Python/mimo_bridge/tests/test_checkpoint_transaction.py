# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Disk errors cannot publish a partial checkpoint or destroy its predecessor."""
import copy
from pathlib import Path

import pytest
import torch

from mimo_bridge.lem_model import LemLite
from mimo_bridge.tools import training_continuation as tc


def test_runtime_failure_keeps_previous_same_step_checkpoint(tmp_path,monkeypatch):
    model=LemLite(device='cpu',imagination=False,no_vision=True,checkpoint_dir=str(tmp_path/'checkpoints'))
    path=Path(model.save_checkpoint(7));original=(path/'runtime_state.pt').read_bytes()
    save=torch.save
    def fail_runtime(value,destination,*args,**kwargs):
        if str(destination).endswith('runtime_state.pt'):
            Path(destination).write_bytes(b'truncated')
            raise RuntimeError('PytorchStreamWriter file write failed')
        return save(value,destination,*args,**kwargs)
    monkeypatch.setattr(torch,'save',fail_runtime)
    with pytest.raises(RuntimeError,match='write failed'):model.save_checkpoint(7)
    assert (path/'runtime_state.pt').read_bytes()==original
    assert list((tmp_path/'checkpoints').glob('ckpt_step_*'))==[path]


def test_physical_companion_failure_never_publishes_new_step(tmp_path):
    model=LemLite(device='cpu',imagination=False,no_vision=True,checkpoint_dir=str(tmp_path/'checkpoints'))
    previous=Path(model.save_checkpoint(7))
    def failure(staging,final):
        assert (staging/'runtime_state.pt').exists() and not final.exists()
        raise OSError('disk full writing physical episode')
    with pytest.raises(OSError,match='physical episode'):model.save_checkpoint(8,companion_writer=failure)
    assert previous.exists() and not (previous.parent/'ckpt_step_8').exists()


def test_modern_checkpoint_cannot_silently_cold_start_missing_or_corrupt_runtime(tmp_path):
    model=LemLite(device='cpu',imagination=False,no_vision=True,checkpoint_dir=str(tmp_path/'checkpoints'))
    path=Path(model.save_checkpoint(7));runtime=path/'runtime_state.pt'
    runtime.write_bytes(b'truncated')
    with pytest.raises(RuntimeError,match='runtime state is corrupt'):model.load_checkpoint(path)
    runtime.unlink()
    with pytest.raises(ValueError,match='required runtime evidence'):model.load_checkpoint(path)


def test_supervisor_skips_corrupt_newest_and_survives_full_log_drive(tmp_path,monkeypatch,capsys):
    from mimo_bridge.tests.test_training_continuation import checkpoint
    old=checkpoint(tmp_path,9);bad=checkpoint(tmp_path,10)
    def validate(path):
        if path==bad:raise RuntimeError('truncated runtime')
    monkeypatch.setattr(tc,'validate_checkpoint',validate)
    selected,rejected=tc.validated_resume_checkpoint(tmp_path,['python'])
    assert selected==old and len(rejected)==1 and rejected[0]['checkpoint']==str(bad)
    def full(*args):raise OSError('disk full')
    monkeypatch.setattr(tc,'atomic_json',full)
    tc.report_state(tmp_path/'status.json',{})
    assert 'continuing supervision' in capsys.readouterr().err


def test_supervisor_follows_declared_external_checkpoint_storage(tmp_path):
    import json
    from mimo_bridge.tests.test_training_continuation import checkpoint
    run=tmp_path/'run';run.mkdir()
    external=tmp_path/'external';expected=checkpoint(external,15)
    (run/'args.json').write_text(json.dumps(dict(checkpoint_storage=str(external/'checkpoints'))))
    assert tc.latest_complete(run)==expected


def test_space_guard_counts_shared_storage_once_and_rejects_full_drive(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from lem.training.checkpoint import require_save_space
    data=torch.zeros(1000000)
    monkeypatch.setattr('shutil.disk_usage',lambda _:SimpleNamespace(free=10**10))
    single=require_save_space(tmp_path,dict(one=data))
    shared=require_save_space(tmp_path,dict(one=data,two=data[:]))
    assert shared['estimated_required_bytes']-single['estimated_required_bytes']<2048
    monkeypatch.setattr('shutil.disk_usage',lambda _:SimpleNamespace(free=1024))
    with pytest.raises(OSError,match='checkpoint storage needs'):require_save_space(tmp_path,dict(data=data))
