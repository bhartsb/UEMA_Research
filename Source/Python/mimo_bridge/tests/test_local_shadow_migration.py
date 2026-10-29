# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from mimo_bridge.event_tape import EventTape
from mimo_bridge.tools.prepare_local_shadow import state_digest,verify_episode


def test_preservation_digest_retains_nan_bytes_paths_and_recorded_event_prefix(tmp_path):
    tape=EventTape(tmp_path/'events',tmp_path/'status.json',reserve_bytes=0)
    tape.root.mkdir()
    tape.append(dict(time=1.,speech='recorded'))
    old=dict(a=np.array([np.nan,1.]),tensor=torch.tensor([2.]),path=Path('known'),events=tape)
    new=copy.deepcopy(old)
    assert state_digest(old)==state_digest(new)
    assert old['events'].tape_id!=new['events'].tape_id
    new['tensor'][0]=3.
    assert state_digest(old)!=state_digest(new)


def test_verifier_refuses_changed_learner_or_world_evidence():
    old=dict(config=dict(contact_profile='mechanical_v2'),learner=dict(memory=torch.ones(2)),
        trainer=dict(step=3),worlds=[dict(env_kwargs=dict(contact_profile='mechanical_v2'),
            world=dict(attrs=dict(_virtual_skin=None,_virtual_skin_transition=None),pose=np.arange(4.)),
            payload=dict(info={},sim_time=3.))])
    new=copy.deepcopy(old);changes=dict(contact_profile='mechanical_v2_local')
    new['config'].update(changes);new['worlds'][0]['env_kwargs'].update(changes)
    new['worlds'][0]['world']['attrs']['_virtual_skin_transition']={'target':'mechanical_v2_local'}
    new['worlds'][0]['payload']['info']['contact_profile_transition']={'target':'mechanical_v2_local'}
    verify_episode(old,new,changes)
    new['worlds'][0]['world']['pose'][0]=9.
    with pytest.raises(ValueError,match='episode evidence'):verify_episode(old,new,changes)


def test_native_prepare_publishes_exact_old_weights_optimizer_and_runtime(tmp_path,monkeypatch):
    """Exercise the actual optimized atomic writer, not a mocked serialization."""
    import json
    from mimo_bridge import continuation_state
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.tools import prepare_local_shadow as migration
    from mimo_bridge.tools.caregiver_learning_trial import verify_loaded
    torch.set_num_threads(1)
    config=dict(contact_profile='mechanical_v2',body_function='modulate',self_choice='off')
    source=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'world'),age=24,
                          caregiver_seconds=0,**{k:v for k,v in config.items() if k!='self_choice'})
    def make():
        return LemLite(device='cpu',imagination=False,no_vision=True,next_frame_loss=True,
                       persistent_memory='on',self_state='learn',self_prediction='grounded_residual',
                       checkpoint_dir=str(tmp_path/'source'/'checkpoints'))
    try:
        old=make();frames=source.reset();cmd=None;dev=old.dev_limits(0)
        for step in range(3):
            cmd=old.step(frames,dev,train=True,step_index=step,executed_action=cmd).motor_for_action
            frames=source.step(cmd)
        old.step_count=3
        checkpoint=Path(old.save_checkpoint(3))
        continuation_state.save(checkpoint,source,old,
            dict(step=3,cmd=cmd,frame_meta=[f.meta for f in frames]),config)
        calibration=tmp_path/'calibration.json'
        calibration.write_text(json.dumps(dict(contact_profile='mechanical_v2_local',production_license=False)))
        monkeypatch.setattr(migration,'native_model',lambda *_:(make(),config))
        target=tmp_path/'prepared'/'checkpoints'/'ckpt_step_3'
        note=migration.prepare(checkpoint,target,calibration,tmp_path/'episodes')
        assert note['old_optimizer_states_exact']>0 and not note['cpu_probe_only']
        assert (checkpoint/'model.safetensors').read_bytes()==(target/'model.safetensors').read_bytes()
        assert (checkpoint/'runtime_state.pt').read_bytes()==(target/'runtime_state.pt').read_bytes()
        loaded=make();info=loaded.load_checkpoint(str(target),load_optimizer=True)
        verify_loaded(loaded,info,checkpoint,True)
        before=continuation_state.load(checkpoint);after=continuation_state.load(target)
        verify_episode(before,after,note['changes'])
        assert not list(target.parent.glob('.tmp_ckpt*'))
    finally:source.close()
