# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from dataclasses import dataclass, field
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from mimo_bridge.skin_contract import SkinLayout
from mimo_bridge.consequence_learning import skin_observation
from lem.model.contact_wellness import contact_response
from lem.model.contact_consequences import ContactConsequences
from mimo_bridge.tools.developmental_trial import confidence


@dataclass
class Frame:
    tokenized: object
    touch_pos: object = None
    touch_sensitivity: object = None
    skin_features: object = None
    meta: dict = field(default_factory=dict)


def test_ue_skin_keeps_sites_beyond_legacy_limit_and_converts_coordinates():
    n = 640
    base = Frame(SimpleNamespace(touch_impulses=torch.zeros(256, 8), motor_capability=torch.ones(10)))
    xyz = np.zeros((n, 3)); xyz[:, 0] = 110
    layout = SkinLayout.from_ue_cm([str(i) for i in range(n)], xyz, [100, 0, 0],
                                   np.diag([1, -1, 1]), np.ones((n, 2)))
    fields = np.zeros((n, 8)); fields[:, 0] = 5000; fields[:, 3] = 1
    fields[-1, 0] = 100000
    observed = skin_observation([layout.attach(base, fields)], 'cpu')
    assert observed['valid'].sum() == n
    assert observed['features'][0, -1, 0] == 1
    torch.testing.assert_close(observed['positions'][0, :, 0], torch.full((n,), .2))
    assert contact_response(observed['features'], observed['sensitivity'], observed['valid'])['pain'] == 1
    bad = layout.attach(base, fields)
    bad.touch_pos = bad.touch_pos[:-1]
    with pytest.raises(ValueError, match='every physical site'):
        skin_observation([bad], 'cpu')


def test_swapping_actuators_is_not_invisible_to_the_predictor():
    torch.manual_seed(29)
    model = ContactConsequences(16, n_handles=2)
    torch.nn.init.normal_(model.delta[-1].weight, std=.1)
    actions = torch.randn(1, 2, 2, 9)
    args = [torch.randn(1, 2, 16), torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool),
            torch.randn(1, 2, 3), torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=torch.bool)]
    a = model(*args, actions, torch.full((1, 2), .05), torch.ones(1, 10))
    b = model(*args, actions.flip(2), torch.full((1, 2), .05), torch.ones(1, 10))
    assert (a['log_features'] - b['log_features']).abs().max() > 1e-5


@pytest.mark.parametrize('values', [[], [1.], [1., float('nan')], [1., float('inf')]])
def test_missing_or_invalid_worlds_cannot_pass(values):
    assert confidence(values)['valid'] is False


def test_caregiver_replay_keeps_trajectory_when_child_looks_elsewhere(tmp_path):
    from pathlib import Path
    from mimo_bridge.datasource import SingleEnvSource
    speech = Path(__file__).resolve().parents[4] / 'Saved/mimo_bridge/caregiver_speech'
    if not (speech / 'manifest.json').exists():
        pytest.skip('build the optional caregiver speech cache for integration')
    kw = dict(scene='crib', vision=False, audio_real=True, mobile=True,
              caregiver_speech_dir=str(speech), gaze_reflex=False, spontaneous_saccades=False)
    source = SingleEnvSource(run_dir=str(tmp_path / 'responding'), caregiver_mode='responsive', **kw)
    positions = []
    try:
        source.reset()
        original = source.env.caregiver.mobile_mount.copy()
        caregiver = source.env.caregiver
        caregiver.state['visible'] = False
        source.env.data.mocap_pos[source.env._target_id] = caregiver.state['anchor'] + [0, .55, .35]
        source.env.model.body_pos[caregiver.mobile_id, 2] += .5
        caregiver.state.update(scene_phase='absent',phase_until=1000)
        caregiver.tick(source.env)
        assert not caregiver.state['object_present'] and not caregiver.state['mobile_present']
        assert all(source.env.model.geom_rgba[g, 3] == 0 for g in caregiver.mobile_geoms)
        assert all(source.env.model.geom_contype[g] == 0 for group in caregiver.object_geoms for g in group)
        source.reset()
        assert all(source.env.model.geom_rgba[g, 3] > 0 for g in caregiver.mobile_geoms)
        for _ in range(60):
            source.env.step(gaze=(0., 0.))
            positions.append((source.env.data.mocap_pos.copy(),source.env.data.mocap_quat.copy(),
                              source.env.model.geom_pos.copy(),source.env.model.geom_quat.copy(),
                              source.env.caregiver.state['mouth_opening']))
        source.env.caregiver.export_events(tmp_path / 'tape.json')
        source.env.model.body_pos[source.env.caregiver.mobile_id, 2] += .4
        source.reset()
        np.testing.assert_array_equal(source.env.model.body_pos[source.env.caregiver.mobile_id], original)
    finally:
        source.close()
    replay = SingleEnvSource(run_dir=str(tmp_path / 'replay'), caregiver_mode='replay',
                             caregiver_replay=str(tmp_path / 'tape.json'), **kw)
    try:
        replay.reset()
        for expected,quats,geom_pos,geom_quat,mouth in positions:
            replay.env.step(gaze=(.5, -.5))
            np.testing.assert_allclose(replay.env.data.mocap_pos, expected, atol=1e-7)
            np.testing.assert_allclose(replay.env.data.mocap_quat,quats,atol=1e-7)
            np.testing.assert_allclose(replay.env.model.geom_pos,geom_pos,atol=1e-7)
            np.testing.assert_allclose(replay.env.model.geom_quat,geom_quat,atol=1e-7)
            assert replay.env.caregiver.state['mouth_opening']==mouth
    finally:
        replay.close()


def test_choice_evaluation_keeps_sensing_during_held_commands(tmp_path, monkeypatch):
    from pathlib import Path
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.tools.developmental_trial import evaluate
    speech = Path(__file__).resolve().parents[4] / 'Saved/mimo_bridge/caregiver_speech'
    if not (speech / 'manifest.json').exists():
        pytest.skip('optional caregiver speech cache is absent')
    kw = dict(device='cpu', no_vision=True, imagination=False,
              overrides={'model': {'context_length': 2}, 'modalities': {'audio': {'data_source': 'real'}}})
    model = LemLite(contact_learning=True, consequence_horizon=2, **kw)
    baseline = LemLite(**kw)
    original_step = model.step
    seen = []
    def observed(frames, *args, **kwargs):
        seen.append((frames[0].sim_time, kwargs.get('executed_action')))
        return original_step(frames, *args, **kwargs)
    monkeypatch.setattr(model, 'step', observed)
    a = dict(no_vision=True, motor_hold=2, fovea_size=0, ring_size=64, _trial_start_step=0)
    evaluate(model, baseline, tmp_path / 'choices', speech, a, model.dev_limits(0),
             [721], decisions=2, horizon=2, repeats=2)
    assert len(seen) == model.ctx_len + 2 * 2
    np.testing.assert_allclose(np.diff([x[0] for x in seen]), .02, atol=1e-9)
    # Same actual body order reaches both observed intervals of the first choice.
    np.testing.assert_array_equal(seen[3][1][:, :model.GAZE_IDX], seen[4][1][:, :model.GAZE_IDX])
