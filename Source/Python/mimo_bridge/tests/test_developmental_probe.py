# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mimo_bridge.tools.causal_test import _limits_for_run
from mimo_bridge.tools.developmental_probe import geometry_for_seed, target_line_of_sight, outcome_declaration, verified_reference, PROFILE_ABILITIES


def test_seed_changes_the_physical_arrangement_and_repeats_exactly():
    assert geometry_for_seed(10) == geometry_for_seed(10)
    assert geometry_for_seed(10) != geometry_for_seed(11)
    worlds = [geometry_for_seed(i) for i in range(20)]
    assert len({row['target_distance_m'] for row in worlds}) == 20
    assert any(row['relocation_m'] < 0 for row in worlds)
    assert any(row['relocation_m'] > 0 for row in worlds)


def test_real_ray_rejects_occlusion_and_absence():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="eye"><camera name="eye_left"/></body>
      <body name="target" pos="0 0 1"><geom name="target_geom" type="sphere" size=".1"/></body>
      <body name="occluder" pos="0 0 .5"><geom type="sphere" size=".15"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    env=SimpleNamespace(model=model,data=data,_mj=mujoco)
    assert target_line_of_sight(env,0,np.array([0,0,1.]),True) is False
    model.body_pos[model.body('occluder').id,0]=1
    mujoco.mj_forward(model,data)
    assert target_line_of_sight(env,0,np.array([0,0,1.]),True) is True
    assert target_line_of_sight(env,0,np.array([0,0,1.]),False) is False


def test_all_four_observation_profiles_name_their_actual_candidate_functions():
    assert set(PROFILE_ABILITIES)=={'toy','body_floor','free_objects','support'}
    assert 'grasp' in PROFILE_ABILITIES['free_objects']
    assert 'stand' in PROFILE_ABILITIES['support']
    assert 'visual_self_recognition' not in set().union(*map(set,PROFILE_ABILITIES.values()))


def no_defaults():
    return SimpleNamespace(dev_limits=lambda _: pytest.fail('An external curriculum checkpoint must not silently use static defaults'))


def make_checkpoint(tmp_path, phase=None):
    external=tmp_path/'external';checkpoint=external/'checkpoints'/'ckpt_step_100';checkpoint.mkdir(parents=True)
    canonical=tmp_path/'canonical';canonical.mkdir()
    (external/'args.json').write_text(json.dumps(dict(out=str(canonical),curriculum='on',age_binding='curriculum')))
    if phase:(checkpoint/'meta.json').write_text(json.dumps(dict(curriculum_state=dict(phase=phase))))
    return checkpoint,canonical


def test_external_checkpoint_resolves_canonical_log_and_nearest_step(tmp_path):
    checkpoint,canonical=make_checkpoint(tmp_path,'mature')
    rows=[dict(type='mimo',step=90,rom_limit=.9,strength_limit=.8,phase='mature',age_months=24),
          dict(type='mimo',step=1000,rom_limit=1.,strength_limit=1.,phase='mature',age_months=24)]
    (canonical/'replay_training_stats.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    limits,source=_limits_for_run(str(checkpoint),no_defaults())
    assert limits['strength_limit']==.8
    assert str(canonical) in source


def test_missing_log_uses_saved_native_curriculum_phase_not_step_schedule(tmp_path):
    checkpoint,_=make_checkpoint(tmp_path,'mature')
    limits,source=_limits_for_run(str(checkpoint),no_defaults())
    assert limits['strength_limit']==1.
    assert limits['age_months']==24.
    assert 'recorded curriculum phase' in source


def test_metadata_phase_prevents_nearby_later_regime_contamination(tmp_path):
    checkpoint,canonical=make_checkpoint(tmp_path,'early_infant')
    (canonical/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo',step=101,phase='mature',rom_limit=1.,strength_limit=1.)))
    limits,_=_limits_for_run(str(checkpoint),no_defaults())
    assert limits['phase_name']=='early_infant'
    assert limits['strength_limit']==.15


def test_external_curriculum_without_any_actual_phase_fails_closed(tmp_path):
    checkpoint,_=make_checkpoint(tmp_path)
    with pytest.raises(ValueError,match='defaults are not evidence'):
        _limits_for_run(str(checkpoint),no_defaults())


def test_reference_true_flag_alone_cannot_license_physical_success(tmp_path):
    receipt=tmp_path/'claim.json';receipt.write_text(json.dumps(dict(reachability_verified=True)))
    d=outcome_declaration('gaze_control',12,.05)
    d['calibration'].update(reachability_verified=True,repeatability_verified=True,receipt=str(receipt))
    assert verified_reference(d,{'body':'same'},42)[0] is False


def test_reference_requires_this_world_condition_and_executed_outcome(tmp_path):
    signature={'body':'same'};receipt=tmp_path/'receipt.json';trace_path=tmp_path/'trace.json'
    receipt.write_text(json.dumps(dict(ability_id='gaze_control',reachability_verified=True,
        same_action_replay_exact=True,body_signature=signature,world_seed=42,reference_trace='trace.json')))
    trace=dict(ability_id='gaze_control',world_seed=42,pair_id='a',phase='acquisition',condition='present',
        arm='baseline',reference_action_source='scripted_reference',time_s=[i*.05 for i in range(13)],
        model_authored=[False]*13,external_assistance=[False]*13,task_valid=[True]*13,
        measurements=dict(target_angle_deg=[0.]*13,target_visible=[True]*13))
    trace_path.write_text(json.dumps(trace))
    d=outcome_declaration('gaze_control',12,.05)
    d['calibration'].update(reachability_verified=True,repeatability_verified=True,receipt=str(receipt))
    assert verified_reference(d,signature,42)[0] is True
    assert verified_reference(d,signature,42,'relocated')[0] is False
    assert verified_reference(d,signature,43)[0] is False
    assert verified_reference(d,{'body':'different'},42)[0] is False
    trace['measurements']['target_angle_deg']=[80.]*13;trace_path.write_text(json.dumps(trace))
    assert verified_reference(d,signature,42)[0] is False
