# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import torch
import pytest
from lem.training.compact_checkpoint import compact_torch_archive
from mimo_bridge.copied_execution import equal_tree


def test_compacted_checkpoint_loads_natively_with_every_tensor_and_history_exact(tmp_path):
    generator=torch.Generator().manual_seed(313)
    shared=torch.randn(10000,generator=generator)
    original={'zeros':torch.zeros(200000),'float':shared,'view':shared[3:17],
        'half':shared.to(torch.float16),'bool':shared>0,'int':torch.arange(300),
        'history':{'step':3491771,'names':['learned','retained'],'none':None}}
    path=tmp_path/'runtime_state.pt';torch.save(original,path)
    report=compact_torch_archive(path)
    actual=torch.load(path,map_location='cpu',weights_only=False)
    assert report['compressed'] and report['stored_bytes']<report['original_bytes']/2
    assert equal_tree(original,actual)
    assert actual['float'].untyped_storage().data_ptr()==actual['view'].untyped_storage().data_ptr()
    actual['float'][3]=123.
    assert actual['view'][0]==123. # Shared storage and view offsets also survive.


def test_invalid_archive_preserves_original_and_cleans_temporary(tmp_path):
    path=tmp_path/'runtime_state.pt';path.write_bytes(b'untouched bad input')
    with pytest.raises(Exception):compact_torch_archive(path)
    assert path.read_bytes()==b'untouched bad input'
    assert not path.with_name(path.name+'.compressing').exists()
