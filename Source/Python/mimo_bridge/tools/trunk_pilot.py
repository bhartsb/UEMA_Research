# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exercise a migrated copy with real sensors, replay and USB caregiver recording."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run
from mimo_bridge.train_mimo import parse_args, env_kwargs_for
from mimo_bridge.datasource import PoolSource
from mimo_bridge.body_layout import extend_limits
from mimo_bridge.prediction_audit import accumulate, summarize


def run(checkpoint, out, recording, steps=70):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model, args, _, _ = _build_model(checkpoint, 'cpu', 0, intention=True)
    model.load_checkpoint(checkpoint, load_optimizer=True)
    model.enable_replay(2, size_per_env=256)
    settings = vars(parse_args(['--out', str(out)]))
    settings.update(args)
    settings.update(caregiver_record_dir=str(recording))
    dev, _ = _limits_for_run(checkpoint, model)
    dev = extend_limits(dev, 26)
    kw = env_kwargs_for(SimpleNamespace(**settings), dev['phase_name'], str(out/'scenes'))
    source = PoolSource(2, kw)
    readings = {}
    replay_losses = []
    skipped_before = model._nonfinite_steps
    try:
        frames = source.reset()
        executed = held = None
        for i in range(steps):
            result = model.step(frames, dev, train=True, executed_action=executed)
            if not np.isfinite(result.loss):
                raise RuntimeError('nonfinite copied-model training loss')
            accumulate(readings, model.last_prediction_audit)
            if held is None or i % 5 == 0:
                held = result.motor_for_action.copy()
            command = held
            gaze_path = []
            for sub in range(5):
                frames = source.step(command, result.gaze_out, dev.get('handle_mask'), render=sub == 4)
                gaze_path.append([f.meta.get('gaze_cmd', (0.,0.)) for f in frames])
            executed = model.gaze_to_handle(command, np.asarray(gaze_path).mean(0))
            if model.replay.count >= model.ctx_len + 3:
                replay = model.replay_update(batch=4)
                if replay is None or not np.isfinite(replay['replay_loss']):
                    raise RuntimeError('copied-model replay update did not complete')
                replay_losses.append(replay['replay_loss'])
        storage = [f.meta.get('caregiver', {}).get('recording') for f in frames]
        if not all(s and not s['error'] and not s['dropped'] for s in storage):
            raise RuntimeError(f'recording failed: {storage}')
        if model._nonfinite_steps != skipped_before:
            raise RuntimeError('copied-model optimizer skipped a nonfinite update')
        if not replay_losses:
            raise RuntimeError('pilot collected too little history to exercise replay')
        result = dict(status='passed_integration', steps=steps, bodies=2,
                      model_parameters=model.n_params, replay_updates=len(replay_losses),
                      replay_losses=replay_losses, nonfinite_updates=0,
                      prediction_audit=summarize(readings), storage=storage,
                      scope='Training, replay, sensing and recording; not a learned behavior test')
        (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return result
    finally:
        source.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--out', required=True)
    ap.add_argument('--recording', required=True)
    ap.add_argument('--steps', type=int, default=70)
    a = ap.parse_args()
    run(a.checkpoint, a.out, a.recording, a.steps)
