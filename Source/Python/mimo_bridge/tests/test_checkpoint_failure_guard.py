# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A failed checkpoint write must not end the run (2026-09-18).

The H84 smoke's crash snapshot failed with "The device is not ready" (os error 21) on a
USB disk whose system log showed eleven retried IO operations and a port reset in six
minutes. Before the guard, the same failure in an HOURLY save raised out of the training
loop; the crash snapshot then failed on the same disk, and the run lost every step since
its last complete checkpoint. Two contracts, driven through the real trainer on CPU with
one body and no vision:

- an hourly save that raises is counted (checkpoint_failures and checkpoint_last_error in
  the stats record), logged, and training continues; the next save is attempted and lands;
- when the configured storage refuses the CRASH snapshot, the snapshot is written to the
  run directory's own checkpoints_fallback folder, so a crash under a disk outage still
  leaves the weights behind.
"""
import json

import torch


def _flags(tmp_path):
    return ['--out', str(tmp_path / 'run'), '--device', 'cpu', '--n-envs', '1', '--no-vision',
            '--curriculum', 'off', '--age-binding', 'none', '--delay-binding', 'off', '--episode-steps', '0',
            '--action-repeat', '1', '--motor-hold', '2', '--minutes', '0', '--max-steps', '3', '--no-babble',
            '--self-state', 'perceive', '--body-function', 'modulate', '--persistent-memory', 'on',
            '--imagination', 'off', '--next-frame-loss', 'on', '--contact-profile', 'mechanical_v2',
            '--replay', 'off', '--report-every', '1', '--video-every-min', '0',
            '--checkpoint-every-min', '0', '--checkpoint-storage', str(tmp_path / 'storage' / 'checkpoints')]


def _isolate(monkeypatch):
    from mimo_bridge.tools import preflight
    monkeypatch.setattr(preflight, 'run_probes', lambda path, env_kw=None: dict(ok=True, checks=[]))
    torch.set_num_threads(1)


def test_a_failed_hourly_save_is_counted_and_training_continues(tmp_path, monkeypatch):
    from mimo_bridge import lem_model
    from mimo_bridge.train_mimo import main
    _isolate(monkeypatch)
    real = lem_model.LemLite.save_checkpoint
    calls = []

    def flaky(self, step, *a, **k):
        calls.append(step)
        if len(calls) == 1:
            raise OSError(21, 'The device is not ready')
        return real(self, step, *a, **k)

    monkeypatch.setattr(lem_model.LemLite, 'save_checkpoint', flaky)
    # --checkpoint-every-min 0: the "hourly" save fires on every step
    assert main(_flags(tmp_path)) == 0
    records = [json.loads(line) for line in (tmp_path / 'run' / 'replay_training_stats.jsonl').read_text().splitlines()]
    mimo = [r for r in records if r['type'] == 'mimo']
    assert mimo[-1]['checkpoint_failures'] == 1
    assert 'device is not ready' in mimo[-1]['checkpoint_last_error']
    assert len(calls) >= 2, 'the save after the failure was never attempted'
    saved = [p.name for p in (tmp_path / 'storage' / 'checkpoints').iterdir() if p.name.startswith('ckpt_step_')]
    assert saved, 'no later save landed'


def test_crash_snapshot_falls_back_to_the_run_directory_when_storage_refuses(tmp_path, monkeypatch):
    from lem.training import checkpoint as CK
    from mimo_bridge import lem_model
    from mimo_bridge.train_mimo import main
    _isolate(monkeypatch)
    storage = str(tmp_path / 'storage')
    real_save = CK.CheckpointManager.save

    def refusing(self, *a, **k):
        if str(self._checkpoint_dir).startswith(storage):
            raise OSError(21, 'The device is not ready')
        return real_save(self, *a, **k)

    monkeypatch.setattr(CK.CheckpointManager, 'save', refusing)
    real_step = lem_model.LemLite.step
    seen = {'calls': 0}

    def crashing(self, *a, **k):
        seen['calls'] += 1
        if seen['calls'] == 2:
            raise RuntimeError('planted crash')
        return real_step(self, *a, **k)

    monkeypatch.setattr(lem_model.LemLite, 'step', crashing)
    flags = _flags(tmp_path)
    flags[flags.index('--checkpoint-every-min') + 1] = '60'      # no hourly save inside three steps
    assert main(flags) == 1
    assert not (tmp_path / 'storage' / 'checkpoints').exists() or not any(
        p.name.startswith('ckpt_step_') for p in (tmp_path / 'storage' / 'checkpoints').iterdir())
    fallback = tmp_path / 'run' / 'checkpoints_fallback'
    assert fallback.exists(), 'no fallback folder'
    assert any(p.name.startswith('ckpt_step_') for p in fallback.iterdir()), 'the fallback snapshot was not written'
