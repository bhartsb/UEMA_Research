# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import hashlib
import json
from pathlib import Path
import pytest
from mimo_bridge.tools import caregiver_learning_trial as trial


def put_fit(folder,steps=9000):
    folder.mkdir(parents=True,exist_ok=True);tape=folder/'adult.json';tape.write_text('[]')
    receipt=dict(online_updates=steps,offline_replay_updates=0,checkpoint=str(folder/'checkpoints'/'ckpt_step_1'),
                 adult_tape=str(tape),source_utterances=[[0.01,'hello']])
    (folder/'fit.json').write_text(json.dumps(receipt))
    return receipt


def test_completed_fit_follows_real_attempt_receipt_and_preserves_incomplete_folder(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,'validate_checkpoint',lambda path:None)
    base=tmp_path/'responsive_7';base.mkdir();partial=base/'partial.txt';partial.write_text('old incomplete learning')
    replacement=trial.next_attempt(base);assert replacement==base/'attempt_002'
    assert trial.completed_fit(base,9000) is None
    data=put_fit(replacement)
    actual=trial.completed_fit(base,9000)
    assert actual[0]==replacement and actual[1]['adult_tape']==data['adult_tape']
    assert partial.read_text()=='old incomplete learning'
    put_fit(base/'attempt_003')
    with pytest.raises(ValueError,match='multiple completed'):trial.completed_fit(base,9000)


def test_original_declaration_and_source_hashes_survive_resume(tmp_path):
    declaration=dict(worlds=12,train_steps=9000,gate='unchanged original gate')
    path=tmp_path/'declaration.json';path.write_text(json.dumps(declaration));before=path.read_bytes()
    source=tmp_path/'source_hashes.json';source.write_text('{"original":"hash"}')
    trial.record_resume_version(tmp_path,{'new':'hash2'},declaration)
    assert path.read_bytes()==before and source.read_text()=='{"original":"hash"}'
    versions=list((tmp_path/'resume_versions').glob('*.json'));assert len(versions)==1
    assert json.loads(versions[0].read_text())['declaration_sha256']==hashlib.sha256(before).hexdigest()
    with pytest.raises(ValueError,match='changed the original'):trial.record_resume_version(tmp_path,{},dict(declaration,train_steps=100))


def test_wrong_learning_budget_and_missing_donor_tape_are_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,'validate_checkpoint',lambda path:None)
    base=tmp_path/'responsive_0';data=put_fit(base,steps=8)
    with pytest.raises(ValueError,match='learning budget'):trial.completed_fit(base,9000)
    Path(data['adult_tape']).unlink()
    with pytest.raises(ValueError,match='actual adult tape'):trial.completed_fit(base,8)


def test_completed_probe_in_replacement_is_reused_without_selecting_new_result(tmp_path):
    base=tmp_path/'probe_follow_visible';base.mkdir();(base/'unfinished.txt').write_text('retained')
    folder=base/'attempt_002';folder.mkdir();result=dict(kind='follow_visible',samples=100,mean_alignment=.6)
    (folder/'result.json').write_text(json.dumps(result))
    assert trial.completed_probe(base,'follow_visible',100)==(folder,result)
    with pytest.raises(ValueError):trial.completed_probe(base,'follow_visible',101)
