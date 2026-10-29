# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The comparison records through the real trainer (train_mimo.main) on the processor: one body, no vision, a small
width, the replay lesson on, the observer recording, 12 model steps (comparison_records.py; design note
docs/gain_of_function_v4/comparison_records/COMPARISON_RECORDS.md).

What it shows: the trainer writes the records' header, one step line per model step and one line per replay pass, the
lane's new columns appear in the observer's model chunks, and a spike event (forced at model step 6, since a run with no
vision grades no vision transition) is written with its inputs and the weights after that model step's update, into
the records' own folder: the run's own checkpoint folder and checkpoints.json do not list it, and the run ends with
exit code 0. And (FIX-2 of the bundle review of Oct 10) every stats record flushes the records: before close, every
step line and every pass line that ran before the last stats record is already on disk."""
import json
import os
from pathlib import Path

import numpy as np
import torch


def _flags(tmp_path):
    return ['--out', str(tmp_path / 'run'), '--device', 'cpu', '--n-envs', '1', '--no-vision',
            '--curriculum', 'off', '--age-binding', 'none', '--delay-binding', 'off', '--episode-steps', '0',
            '--action-repeat', '1', '--motor-hold', '2', '--minutes', '0', '--max-steps', '12', '--no-babble',
            '--self-state', 'perceive', '--body-function', 'modulate', '--persistent-memory', 'on',
            '--imagination', 'off', '--next-frame-loss', 'on', '--contact-profile', 'mechanical_v2',
            '--replay', 'on', '--replay-every', '2', '--replay-batch', '4', '--replay-size', '64',
            '--context-length', '4', '--d-model', '32', '--core-blocks', '1', '--encoder-width', '32',
            '--encoder-layers', '1', '--observer', 'record', '--observer-chunk-steps', '4',
            '--observer-reserve-gb', '1', '--report-every', '4', '--video-every-min', '0',
            '--checkpoint-every-min', '600', '--checkpoint-storage', str(tmp_path / 'storage' / 'checkpoints')]


def test_the_trainer_writes_every_record_and_keeps_a_spike_events_weights_in_the_records_folder(tmp_path, monkeypatch):
    from mimo_bridge import comparison_records as CR
    from mimo_bridge.observer_model_lane import SENSES, read_model_chunk
    from mimo_bridge.tools import preflight
    from mimo_bridge.train_mimo import main
    monkeypatch.setattr(preflight, 'run_probes', lambda path, env_kw=None: dict(ok=True, checks=[]))
    torch.set_num_threads(1)
    real = CR.ComparisonRecords.after_step

    def forced(self, step, model, res, snapshot=None):
        if step == 6:                           # a forced event: this run grades no vision transition
            self.spike.pending = dict(step=6, bodies=[0], first_step=6,
                                      captures=[(6, dict(note=torch.arange(3.0)))])
        return real(self, step, model, res, snapshot)
    monkeypatch.setattr(CR.ComparisonRecords, 'after_step', forced)
    # FIX-2: what is on disk just before close() is what a stop that kills the process would leave (learning_on's
    # stops end the process tree, so no close runs)
    seen = {}
    real_close = CR.ComparisonRecords.close

    def close(self):
        for name, w in (('steps', self.steps), ('passes', self.passes)):
            seen[name] = ([json.loads(x) for x in open(w.path, encoding='utf-8').read().splitlines()]
                          if os.path.isfile(w.path) else [])
        seen['flushes'] = self.flushes
        return real_close(self)
    monkeypatch.setattr(CR.ComparisonRecords, 'close', close)
    assert main(_flags(tmp_path)) == 0
    folder = tmp_path / 'run' / 'comparison_records'
    header = json.loads(sorted(folder.glob('header_*.json'))[0].read_text(encoding='utf-8'))
    assert header['version'] == CR.RECORDS_VERSION and header['spike']['line'] == CR.SPIKE_LINE
    steps = [json.loads(x) for x in sorted(folder.glob('steps_*.jsonl'))[0].read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in steps] == list(range(12))
    assert all('core' in r['group_norms'] for r in steps if r['grad_norm'])
    passes = [json.loads(x) for x in
              sorted(folder.glob('replay_passes_*.jsonl'))[0].read_text(encoding='utf-8').splitlines()]
    assert passes and all(p['step'] % 2 == 0 for p in passes)
    for p in passes:
        w = p['windows']
        assert set(w['body']) == {0} and all(t - f == 4 for f, t in zip(w['first'], w['target']))
        assert all(t <= p['step'] - 1 for t in w['target'])     # the pass runs after model step step - 1
    # the lane's new columns
    z = [read_model_chunk(str(c)) for c in sorted((tmp_path / 'run' / 'observer' / 'model').glob('chunk_*.npz'))]
    pers = np.concatenate([c['persist_pe_body'] for c in z])
    assert pers.shape == (12, 1, len(SENSES))
    assert np.isfinite(pers[1:, :, SENSES.index('proprioception')]).all()
    assert np.isnan(np.concatenate([c['vision_no_copy_body'] for c in z])).all()     # no vision in this run
    # the forced event: inputs and weights in the records' folder, none in the run's own checkpoints
    ev = [json.loads(x) for x in
          sorted((folder / 'vision_spike').glob('events_*.jsonl'))[0].read_text(encoding='utf-8').splitlines()]
    assert len(ev) == 1 and ev[0]['step'] == 6 and ev[0]['captured_steps'] == [6]
    ck = folder / 'vision_spike' / 'checkpoints' / header['launch'] / 'ckpt_step_6'
    assert Path(ev[0]['weights']) == ck, (ev[0]['weights'], ck)
    assert (ck / 'model.safetensors').is_file() and (ck / 'optimizer.pt').is_file()
    # NOTE-4: the inputs file carries the launch in its name and no temporary file is left
    assert Path(ev[0]['inputs']).name == 'inputs_step_6_%s.pt' % header['launch']
    assert not list((folder / 'vision_spike').glob('*.part'))
    assert not (ck / 'episode_state.json').exists(), 'the event keeps the weights only, no episode companion'
    data = torch.load(ev[0]['inputs'], weights_only=False)
    assert torch.equal(data['transitions'][0]['note'], torch.arange(3.0))
    listed = json.loads((tmp_path / 'run' / 'checkpoints.json').read_text(encoding='utf-8'))
    assert listed and not [e for e in listed if 'vision_spike' in str(e.get('path'))]
    saved = sorted(p.name for p in (tmp_path / 'storage' / 'checkpoints').iterdir() if p.name.startswith('ckpt_'))
    assert saved == ['ckpt_step_12'], saved
    status = json.loads(sorted(folder.glob('status_*.json'))[0].read_text(encoding='utf-8'))
    assert status['steps']['written'] == 12 and status['steps']['error'] is None
    # FIX-2: the stats records at counters 4, 8 and 12 (--report-every 4) each flushed the records before close: every
    # step line (model steps 0 to 11) and every pass line that ran before the last record (counters below 12) was on
    # disk; only the pass of counter 12, which runs after that record, was still held
    assert seen['flushes'] == 3 and status['flushes'] == 4
    assert [r['step'] for r in seen['steps']] == list(range(12))
    assert seen['passes'] and [p['step'] for p in seen['passes']] == [p['step'] for p in passes if p['step'] < 12]
