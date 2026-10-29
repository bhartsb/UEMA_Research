# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from pathlib import Path

import pytest
import torch

from mimo_bridge.tools.lossless_record_archive import prepare_archive, restore_archive


def test_full_native_record_restores_without_dropping_runtime(tmp_path):
    source = tmp_path / 'record.pt'
    original = {'execution_output': {'weights': torch.arange(40), 'history': [4, 8]},
                'forecast': torch.tensor([1.5, 2.5])}
    torch.save(original, source)
    before = source.read_bytes()
    record = prepare_archive(source)
    assert source.read_bytes() == before
    assert record['all_original_bytes_preserved']
    restored = restore_archive(tmp_path / 'record.pt.archive.json', tmp_path / 'copy.pt')
    assert restored.read_bytes() == before
    loaded = torch.load(restored, weights_only=False)
    assert torch.equal(loaded['execution_output']['weights'], original['execution_output']['weights'])
    assert loaded['execution_output']['history'] == [4, 8]
    assert prepare_archive(source)['original_sha256'] == record['original_sha256']


def test_corrupt_archive_cannot_publish_a_restored_record(tmp_path):
    source = tmp_path / 'record.pt'
    source.write_bytes(b'full evidence' * 1000)
    prepare_archive(source)
    archive = tmp_path / 'record.pt.xz'
    archive.write_bytes(archive.read_bytes()[:-4])
    with pytest.raises(ValueError):
        restore_archive(tmp_path / 'record.pt.archive.json', tmp_path / 'copy.pt')
    assert not (tmp_path / 'copy.pt').exists()
    assert source.read_bytes() == b'full evidence' * 1000


def test_restore_never_overwrites_unrelated_destination(tmp_path):
    source = tmp_path / 'record.pt'
    source.write_bytes(b'full evidence')
    prepare_archive(source)
    destination = tmp_path / 'copy.pt'
    destination.write_bytes(b'different evidence')
    with pytest.raises(FileExistsError):
        restore_archive(tmp_path / 'record.pt.archive.json', destination)
    assert destination.read_bytes() == b'different evidence'
