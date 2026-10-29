# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from pathlib import Path
from mimo_bridge.tools.causal_test import _build_model
from mimo_bridge.tools.prepare_continuity import prepare
from lem.training.checkpoint import CheckpointManager


def test_migration_preserves_developmental_stage_and_other_checkpoint_metadata(tmp_path):
    root=tmp_path/'old';root.mkdir()
    (root/'args.json').write_text(json.dumps(dict(context_length=4,next_frame_loss='on',imagination='off',curriculum='on')),encoding='utf-8')
    source=root/'checkpoints/ckpt_step_17'
    old,_,_,_=_build_model(str(source),'cpu',0,intention=True,load=False)
    old.checkpoints=CheckpointManager(old.config,str(source.parent));old.save_checkpoint(17)
    meta=json.loads((source/'meta.json').read_text())
    meta.update(curriculum_state=dict(phase='mature',phase_step=734),extra_training_state=dict(example=42))
    (source/'meta.json').write_text(json.dumps(meta),encoding='utf-8')
    (source/'lineage.json').write_text(json.dumps(dict(phase='mature',parent='located_parent')),encoding='utf-8')
    target=tmp_path/'new/checkpoints/ckpt_step_17'
    receipt=prepare(source,target)
    for name in ('meta.json','lineage.json','runtime_state.pt'):
        assert (source/name).read_bytes()==(target/name).read_bytes()
    restored,_,_,_=_build_model(str(target),'cpu',0,intention=True,load=False)
    loaded=restored.load_checkpoint(str(target))
    assert loaded['curriculum_state']==meta['curriculum_state']
    assert receipt['curriculum_state']['phase']=='mature'
    assert receipt['checkpoint_metadata_exact']
