# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Learning the movement map must preserve the controller through a planning event.

Unlike the earlier six-step check, each model carries its own continuing random
stream. This catches extra random draws as well as unintended gradient coupling.
"""
import numpy as np
import pytest
import torch

from mimo_bridge.tests.test_gof_b5_checkpoint import H84, _build, _world, _dev

# CKPT is None when the checkpoint is nowhere; H84.mark() skips then and names every place searched
pytestmark = [pytest.mark.slow, H84.mark()]


def test_learning_map_preserves_orders_weights_and_randomness_through_planning(tmp_path):
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(4)
    args, off, _ = _build(tmp_path)
    _, on, _ = _build(tmp_path, motor_synergy='on', synergy_planner='off', synergy_babble='off')
    before = {k: v.clone() for k, v in on.synergy.state_dict().items()}
    streams = [torch.get_rng_state().clone() for _ in range(2)]
    src = _world(args, tmp_path)
    prev = prev_id = held = None
    seq = 0
    planned = False
    try:
        frames = src.reset()
        for m in (off, on):m.reset_context(1)
        for i in range(26):
            results = []
            step = 344400 + i
            for j, m in enumerate((off, on)):
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(streams[j])
                    results.append(m.step(frames, _dev(args, m), train=True, step_index=step,
                                          executed_action=prev, executed_order_id=prev_id))
                    streams[j] = torch.get_rng_state().clone()
            assert torch.equal(streams[0], streams[1]), f'extra random draws at step {i}'
            assert np.array_equal(results[0].motor_for_action, results[1].motor_for_action), i
            assert torch.equal(off.noise_rng.get_state(), on.noise_rng.get_state()), i
            planned |= isinstance(off.last_intention, dict) and off.last_intention.get('step') == step
            if step % args.motor_hold == 0 or held is None:
                held, seq = results[0].motor_for_action, seq + 1
            on._syn_execution = {'order_id': seq, 'source': 'controller_test'}
            gaze = []
            for repeat in range(args.action_repeat):
                frames = src.step(held, results[0].gaze_out, None, render=repeat == args.action_repeat - 1)
                gaze.append(np.asarray(frames[0].meta.get('gaze_cmd') or (0., 0.), dtype=np.float32))
            prev = LemLite.gaze_to_handle(held, np.mean(np.stack(gaze), axis=0)[None])
            prev_id = seq
    finally:
        src.close()
    assert planned, 'the comparison never exercised planning'
    for a, b in zip(off.params, on.params):
        assert torch.equal(a, b), 'the map lesson changed an existing weight'
    assert any(not torch.equal(before[k], v) for k, v in on.synergy.state_dict().items())
    stats = on.synergy_window_summary()
    assert stats['orders'] >= 4
    assert stats['following_by_author']['controller_test']['part_orders'] > 0
