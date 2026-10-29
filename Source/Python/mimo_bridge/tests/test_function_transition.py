# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import numpy as np
import pytest
from mimo_bridge.function_transition import activate_function
from mimo_bridge.datasource import SingleEnvSource


def test_activation_keeps_last_experience_and_body_then_changes_real_operator(tmp_path):
    src=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path),age=24,
        caregiver_seconds=0,body_function='observe',contact_profile='mechanical_v2')
    restored=None
    try:
        frames=src.reset()
        command=np.zeros((1,len(frames[0].bones),9),np.float32)
        command[0,:,:3]=frames[0].bones[:,4:7]
        command[0,:,3:7]=frames[0].bones[:,:4]
        for _ in range(3):frames=src.step(command)
        saved=src.save_full_state()
        original=dict(config=dict(body_function='observe',self_state='learn'),worlds=saved,
            learner=dict(remembered=np.arange(5)),trainer=dict(step=17,frame_meta=[frames[0].meta]))
        old_quality=saved[0]['world']['attrs']['body_function'].quality.copy()
        upgraded=activate_function(original)
        assert original['config']['body_function']=='observe'
        assert upgraded['config']['body_function']=='modulate'
        np.testing.assert_array_equal(original['worlds'][0]['world']['attrs']['body_function'].quality,old_quality)
        np.testing.assert_array_equal(upgraded['learner']['remembered'],original['learner']['remembered'])
        np.testing.assert_equal(upgraded['trainer']['frame_meta'],original['trainer']['frame_meta'])
        assert upgraded['worlds'][0]['payload']['info']['body_function']['mode']=='observe'
        restored=SingleEnvSource(**upgraded['worlds'][0]['env_kwargs'])
        before=restored.restore_full_state(upgraded['worlds'])[0]
        np.testing.assert_array_equal(before.bones,frames[0].bones)
        np.testing.assert_array_equal(before.patches,frames[0].patches)
        assert before.sim_time==frames[0].sim_time
        after=restored.step(command)[0]
        assert after.meta['body_function']['mode']=='modulate'
        assert after.meta['motor_function']['mode']=='modulate'
        assert after.sim_time>before.sim_time
        np.testing.assert_array_equal(after.patches,restored.env.body_function.observe(after.meta['physical_touch_audit']))
        # A second migration cannot silently erase already experienced impairment.
        with pytest.raises(ValueError,match='observation-only'):activate_function(upgraded)
    finally:
        src.close()
        if restored is not None:restored.close()


def test_growth_outputs_only_inside_callers_directory(tmp_path,monkeypatch):
    from mimo_bridge.env import grow_scene
    import mimo_infant.growth.growth as growth
    import xml.etree.ElementTree as ET
    external=tmp_path/'source';external.mkdir()
    model=external/'MIMo_modelv2.xml';model.write_text('<mujoco>\n<body name="example"/>\n</mujoco>')
    meta=external/'MIMo_metav2.xml';meta.write_text('<mujoco>\n<actuator/>\n</mujoco>')
    source=external/'scene.xml';source.write_text('<mujoco><include file="MIMo_modelv2.xml"/><include file="MIMo_metav2.xml"/></mujoco>')
    originals={p:p.read_bytes() for p in external.iterdir()}
    target=tmp_path/'run'/'grown.xml'
    def upstream(age,scene,*args,**kwargs):
        from pathlib import Path
        scene=Path(scene)
        assert scene.is_relative_to(target.parent)
        for item in ET.parse(scene).getroot().iter('include'):
            assert Path(item.get('file')).is_relative_to(target.parent)
        result=scene.with_name('result.xml');result.write_text('<mujoco/>')
        return str(result)
    monkeypatch.setattr(growth,'adjust_mimo_to_age',upstream)
    assert grow_scene(str(source),24,str(target))==str(target)
    assert target.read_text()=='<mujoco/>'
    assert {p:p.read_bytes() for p in external.iterdir()}==originals
