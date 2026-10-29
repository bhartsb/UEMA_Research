# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch

from mimo_bridge.speech import SpeechStream
from mimo_bridge.audio_adapter import CribAudio, render_track
from lem.model.spatial import vision_regions
from lem.model.contact_consequences import ContactConsequences


def test_speech_changes_with_time_and_video_uses_same_ear_waveform():
    a = CribAudio(.01)
    sr = 16000
    t = np.arange(sr // 4) / sr
    clip = np.concatenate([.2 * np.sin(2 * np.pi * 350 * t),
                           .2 * np.sin(2 * np.pi * 1400 * t)])
    a.speech.start(clip, [.3, .1, 0])
    spectra, frames = [], []
    for _ in range(60):
        spectra.append(a.render([0, 0, 0], [0, 1, 0], [1, 0, 0])[0])
        frames.append(copy.deepcopy(a.last_sources))
    freqs = np.geomspace(40, 8000, 256)
    assert abs(freqs[np.argmax(spectra[15][:, 0])] - 350) < 30
    assert abs(freqs[np.argmax(spectra[40][:, 0])] - 1400) < 100
    pcm = np.concatenate([f['speech_pcm'] for f in frames])
    np.testing.assert_allclose(render_track(frames, .01).astype(float) / 32767, pcm, atol=1/32767)
    assert not a.speech.busy
    assert np.max(np.abs(frames[-1]['speech_pcm'])) == 0


def test_speech_time_and_ear_sign_survive_snapshot():
    a = CribAudio(.013)
    a.speech.start(np.ones(16000) * .1, [.3, .2, 0])
    before = copy.deepcopy(a)
    s, meta = a.render([0, 0, 0], [0, 1, 0], [1, 0, 0])
    s2, meta2 = before.render([0, 0, 0], [0, 1, 0], [1, 0, 0])
    np.testing.assert_array_equal(s, s2)
    assert meta == meta2
    assert meta['ild_db'] < 0 and meta['itd_ms'] < 0


def test_vision_preserves_eye_and_location_after_pooling():
    angle = torch.tensor([[[-1., -1., -1., .1], [1., 1., -1., .1],
                            [-1., -1., 1., .1], [1., 1., 1., .1]]])
    x = torch.eye(4).unsqueeze(0)
    regions, valid = vision_regions(x, angle, grid=2)
    assert valid.sum() == 4
    assert torch.unique(regions[valid], dim=0).shape[0] == 4
    perm = torch.tensor([2, 0, 3, 1])
    y, v = vision_regions(x[:, perm], angle[:, perm], grid=2)
    torch.testing.assert_close(regions, y)
    torch.testing.assert_close(valid, v)


def test_contact_head_learns_an_action_effect_and_keeps_site_identity():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    model = ContactConsequences(16)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    b, n = 16, 2
    context = torch.zeros(b, 2, 16)
    spatial = torch.zeros(b, 1, 16)
    positions = torch.zeros(b, n, 3)
    positions[:, 0, 0] = -.5
    positions[:, 1, 0] = .5
    features = torch.zeros(b, n, 8)
    valid = torch.ones(b, n, dtype=torch.bool)
    actions = torch.zeros(b, 2, 1, 9)
    actions[:8, :, :, 0] = -.5
    actions[8:, :, :, 0] = .5
    targets = features.clone()
    targets[:8, 0, 0] = .8
    targets[8:, 1, 0] = .8
    cap = torch.ones(b, 10)
    for _ in range(120):
        pred = model(context, spatial, valid[:, :1], positions, features, valid,
                     actions, torch.full((b, 2), .05), cap)
        loss, _ = model.loss(pred, targets, valid, cap, cap)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        predicted = model(context, spatial, valid[:, :1], positions, features, valid,
                          actions, torch.full((b, 2), .05), cap)['features']
    assert (predicted[:8, 0, 0] > predicted[:8, 1, 0] + .3).all()
    assert (predicted[8:, 1, 0] > predicted[8:, 0, 0] + .3).all()


def test_live_caregiver_and_delayed_lesson_roundtrip(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    speech_dir = Path(__file__).resolve().parents[4] / 'Saved/mimo_bridge/caregiver_speech'
    if not (speech_dir / 'manifest.json').exists():
        pytest.skip('run Scripts/build_caregiver_speech.py for the real TTS integration test')
    source = SingleEnvSource(scene='crib', run_dir=str(tmp_path / 'scene'),
                             vision=True, eye_size=64, audio_real=True,
                             caregiver_mode='responsive', caregiver_speech_dir=str(speech_dir),
                             touch_geometry='surface',
                             gaze_reflex=False, spontaneous_saccades=False)
    try:
        model = LemLite(device='cpu', contact_learning=True, consequence_horizon=2,
                        next_frame_loss=True, intention=True, imagination=False,
                        overrides={'model': {'context_length': 2},
                                   'modalities': {'audio': {'data_source': 'real'}}})
        frames = source.reset()
        command = None
        initial = model.contact_consequences.delta[-1].weight.detach().clone()
        for step in range(5):
            result = model.step(frames, model.dev_limits(step), train=True, step_index=step,
                                executed_action=command)
            command = result.motor_for_action
            frames = source.step(command, result.gaze_out)
        assert not torch.equal(initial, model.contact_consequences.delta[-1].weight)
        assert 'skin_log_mse' in model.consequence_lesson.stats
        assert np.isfinite(result.loss)
        assert model.contact_consequences in model._side_modules().values()
        ids = {id(p) for g in model.optimizer.param_groups for p in g['params']}
        assert all(id(p) in ids for p in model.contact_consequences.parameters())
        # The delayed bodily lesson must train the shared representation, not
        # merely a detached side predictor. Check its own loss in isolation.
        from lem.training.checkpoint import CheckpointManager
        lesson = model.consequence_lesson
        c = lesson.current
        hidden, spatial, valid = lesson._state(model, c['frame'], c['mask'], c['context'], c['skin'])
        from mimo_bridge.lem_model import command_for_encoder
        prediction = model.contact_consequences(hidden, spatial, valid, c['skin']['positions'],
            c['skin']['features'], c['skin']['valid'],
            command_for_encoder(torch.as_tensor(command))[:, None], torch.full((1, 1), .01), c['skin']['capacity'])
        target = c['skin']['features'].clone(); target[..., 0] += .2
        loss, _ = model.contact_consequences.loss(prediction, target, c['skin']['valid'],
                                                  c['skin']['capacity'], c['skin']['capacity'])
        parameters = list(model.encoder_registry.parameters()) + list(model.forward_model.parameters())
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        split = len(list(model.encoder_registry.parameters()))
        assert any(g is not None and g.abs().sum() > 0 for g in grads[:split])
        assert any(g is not None and g.abs().sum() > 0 for g in grads[split:])
        model.checkpoints = CheckpointManager(model.config, str(tmp_path / 'checkpoints'))
        path = model.save_checkpoint(5)
        expected = {k: v.clone() for k, v in model.contact_consequences.state_dict().items()}
        with torch.no_grad():
            for p in model.contact_consequences.parameters(): p.add_(1)
        model.load_checkpoint(path, load_optimizer=True)
        for k, v in model.contact_consequences.state_dict().items():
            torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
        snap = source.env.save_full_state()
        a = source.env.step(motor=command[0])
        source.env.restore_full_state(snap)
        b = source.env.step(motor=command[0])
        np.testing.assert_allclose(a.info['audio_spec'], b.info['audio_spec'], atol=1e-7)
        np.testing.assert_allclose(a.proprio.bones, b.proprio.bones, atol=1e-7)
    finally:
        source.close()
