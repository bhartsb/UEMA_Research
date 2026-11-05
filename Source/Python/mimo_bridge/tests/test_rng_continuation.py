# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""How a saved CUDA random-number state continues on a device without CUDA (Apple port work item WI-11).

The released episode companion (weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025/
episode_state.pt.gz, 621,243,950 bytes) was written by a cuda:0 run, so its learner state carries
torch.cuda.get_rng_state_all(). Every test here runs on the cpu device passed explicitly, with a small
learner-shaped stand-in and a hand-built state, never the 621 MB companion.
"""
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge import acting_snapshot, continuation_state
from mimo_bridge.continuation_state import (CUDA_RNG_DROPPED, ExactRngContinuation, ReseedRngContinuation,
                                             cuda_rng_seed, restore_learner, rng_continuation)

EXACT_MESSAGE = 'exact continuation requires the saved execution device'
CPU = torch.device('cpu')
# checkout-relative: this file sits at <checkout>/Source/Python/mimo_bridge/tests, so parents[4] is the checkout
RELEASED_RUN = Path(__file__).resolve().parents[4] / 'weights' / 'b1_continue_B1K_skip13_492800' / 'run'


def fake_cuda_state(seed=7, generators=1):
    """What torch.cuda.get_rng_state_all() returns: one uint8 tensor per CUDA device (16 bytes on torch 2.6)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (16,), generator=g, dtype=torch.uint8) for _ in range(generators)]


def fake_model():
    """The attributes restore_learner touches, with nothing else of a learner."""
    return SimpleNamespace(device=CPU, memory_history=None, self_history=None, self_rehearsal=None,
                           loss_manager=SimpleNamespace(_vicreg_history=None),
                           placeholders=SimpleNamespace(_generators={}),
                           noise_rng=torch.Generator())


def learner_state(cuda_rng):
    """A capture_learner-shaped dict with known streams: noise_rng seeded 11, torch seeded 12, numpy 13, python 14."""
    noise = torch.Generator().manual_seed(11)
    torch.manual_seed(12)
    np.random.seed(13)
    random.seed(14)
    return dict(data={}, objects={}, memory=None, self_history=None, self_rehearsal=None, loss_history=None,
                placeholders={}, noise_rng=noise.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=cuda_rng,
                numpy_rng=np.random.get_state(), python_rng=random.getstate())


def actor_rng_state(cuda):
    """What acting_snapshot.rng_state wrote: torch seeded 21, numpy 22, python 23, and the given cuda entry."""
    torch.manual_seed(21)
    np.random.seed(22)
    random.seed(23)
    return dict(torch=torch.get_rng_state().clone(), numpy=np.random.get_state(), python=random.getstate(), cuda=cuda)


def test_cuda_rng_seed_is_the_sha256_of_the_bytes_in_list_order():
    state = fake_cuda_state(generators=2)
    seed, hexdigest, nbytes = cuda_rng_seed(state)
    import hashlib
    expected = hashlib.sha256(b''.join(bytes(v.numpy()) for v in state)).hexdigest()
    assert hexdigest == expected
    assert nbytes == 32
    assert seed == int(expected[:16], 16)
    assert 0 <= seed <= 2 ** 64 - 1
    assert cuda_rng_seed(fake_cuda_state(seed=8))[0] != seed


def test_rng_continuation_names_each_policy_and_refuses_other_words():
    assert isinstance(rng_continuation(None), ExactRngContinuation)
    assert isinstance(rng_continuation('exact'), ExactRngContinuation)
    assert isinstance(rng_continuation('reseed'), ReseedRngContinuation)
    policy = ReseedRngContinuation()
    assert rng_continuation(policy) is policy
    with pytest.raises(ValueError, match='exact'):
        rng_continuation('cuda')
    assert continuation_state.RNG_CONTINUATION_MODES == ('exact', 'reseed')


def test_exact_raises_on_cpu_when_a_cuda_state_was_saved():
    model = fake_model()
    state = learner_state(fake_cuda_state())
    with pytest.raises(ValueError, match=EXACT_MESSAGE):
        restore_learner(model, state)
    with pytest.raises(ValueError, match=EXACT_MESSAGE):
        restore_learner(model, state, rng='exact')
    with pytest.raises(ValueError, match=EXACT_MESSAGE):
        ExactRngContinuation().adopt(fake_cuda_state(), CPU)


def test_reseed_restores_on_cpu_logs_the_drop_and_seeds_torch_from_the_bytes(caplog):
    model = fake_model()
    cuda = fake_cuda_state()
    state = learner_state(cuda)
    seed, hexdigest, nbytes = cuda_rng_seed(cuda)
    with caplog.at_level(logging.WARNING, logger='mimo_bridge.continuation_state'):
        restore_learner(model, state, rng='reseed')
    records = [r for r in caplog.records if CUDA_RNG_DROPPED in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith('restore_learner: ' + CUDA_RNG_DROPPED)
    assert hexdigest in message and str(seed) in message and f'{nbytes} bytes' in message
    assert "differs from the saved run's" in message
    # the torch CPU generator is the one torch.manual_seed(seed) leaves, not the saved torch_rng
    expected = torch.Generator().manual_seed(seed).get_state()
    assert torch.equal(torch.get_rng_state(), expected)
    assert not torch.equal(torch.get_rng_state(), state['torch_rng'])
    # the other streams are the saved ones
    assert torch.equal(model.noise_rng.get_state(), state['noise_rng'])
    assert random.getstate() == state['python_rng']
    assert np.random.get_state()[1].tolist() == state['numpy_rng'][1].tolist()


def test_reseed_is_repeatable_from_the_same_bytes_and_differs_between_states():
    draws = []
    for cuda in (fake_cuda_state(seed=1), fake_cuda_state(seed=1), fake_cuda_state(seed=2)):
        restore_learner(fake_model(), learner_state(cuda), rng='reseed')
        draws.append(torch.rand(4).tolist())
    assert draws[0] == draws[1]
    assert draws[0] != draws[2]


def test_cuda_rng_none_restores_under_both_policies():
    for rng in (None, 'exact', 'reseed'):
        model = fake_model()
        state = learner_state(None)
        torch.manual_seed(99)
        restore_learner(model, state, rng=rng)
        assert torch.equal(torch.get_rng_state(), state['torch_rng'])
        assert torch.equal(model.noise_rng.get_state(), state['noise_rng'])
        assert random.getstate() == state['python_rng']


def test_reseed_restores_exactly_on_a_cuda_device_without_logging(caplog):
    """On CUDA the policy makes no difference: adopt returns the saved tensors moved to cpu for set_rng_state_all."""
    cuda = fake_cuda_state()
    with caplog.at_level(logging.WARNING, logger='mimo_bridge.continuation_state'):
        adopted = ReseedRngContinuation().adopt(cuda, torch.device('cuda'))
    assert [v.tolist() for v in adopted] == [v.tolist() for v in cuda]
    assert not [r for r in caplog.records if CUDA_RNG_DROPPED in r.getMessage()]
    assert [v.tolist() for v in ExactRngContinuation().adopt(cuda, torch.device('cuda'))] == [v.tolist() for v in cuda]


def test_reseed_warning_goes_to_the_injected_logger(caplog):
    log = logging.getLogger('test_rng_continuation.injected')
    with caplog.at_level(logging.WARNING, logger=log.name):
        rng_continuation('reseed', log=log).adopt(fake_cuda_state(), CPU, source='here')
    assert [r.name for r in caplog.records if CUDA_RNG_DROPPED in r.getMessage()] == [log.name]
    assert caplog.records[0].getMessage().startswith('here: ' + CUDA_RNG_DROPPED)


def test_acting_snapshot_restore_rng_has_the_same_guard(caplog):
    state = actor_rng_state(fake_cuda_state())
    with pytest.raises(ValueError, match=EXACT_MESSAGE):
        acting_snapshot.restore_rng(state, CPU)
    with pytest.raises(ValueError, match=EXACT_MESSAGE):
        acting_snapshot.restore_rng(state, CPU, rng='exact')
    seed = cuda_rng_seed(state['cuda'])[0]
    with caplog.at_level(logging.WARNING, logger='mimo_bridge.continuation_state'):
        acting_snapshot.restore_rng(state, CPU, rng='reseed')
    messages = [r.getMessage() for r in caplog.records if CUDA_RNG_DROPPED in r.getMessage()]
    assert len(messages) == 1 and messages[0].startswith('acting snapshot random: ' + CUDA_RNG_DROPPED)
    assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(seed).get_state())
    assert random.getstate() == state['python']
    # no saved CUDA state: both policies restore the saved torch stream
    for rng in (None, 'reseed'):
        plain = actor_rng_state(None)
        torch.manual_seed(5)
        acting_snapshot.restore_rng(plain, CPU, rng=rng)
        assert torch.equal(torch.get_rng_state(), plain['torch'])


def test_adopt_rng_drops_the_cuda_entry_once_so_later_restores_do_not_reseed_again(caplog):
    state = actor_rng_state(fake_cuda_state())
    seed = cuda_rng_seed(state['cuda'])[0]
    before = acting_snapshot.rng_state(CPU)
    with caplog.at_level(logging.WARNING, logger='mimo_bridge.continuation_state'):
        adopted = acting_snapshot.adopt_rng(state, CPU, rng='reseed')
    assert adopted['cuda'] is None
    assert torch.equal(adopted['torch'], torch.Generator().manual_seed(seed).get_state())
    # the process's own stream is left as it was (preserve_rng)
    assert torch.equal(torch.get_rng_state(), before['torch'])
    assert len([r for r in caplog.records if CUDA_RNG_DROPPED in r.getMessage()]) == 1
    caplog.clear()
    acting_snapshot.restore_rng(adopted, CPU)          # exact: nothing to drop any more
    assert torch.rand(1).item() == torch.rand(1, generator=torch.Generator().manual_seed(seed)).item()
    assert not [r for r in caplog.records if CUDA_RNG_DROPPED in r.getMessage()]
    with pytest.raises(ValueError, match=EXACT_MESSAGE):
        acting_snapshot.adopt_rng(state, CPU)


def test_trainer_flag_defaults_to_exact_and_stays_out_of_the_acting_configuration():
    from mimo_bridge import train_mimo
    args = train_mimo.parse_args(['--out', 'unused'])
    assert args.rng_continuation == 'exact'
    assert train_mimo.parse_args(['--out', 'unused', '--rng-continuation', 'reseed']).rng_continuation == 'reseed'
    with pytest.raises(SystemExit):
        train_mimo.parse_args(['--out', 'unused', '--rng-continuation', 'cuda'])
    # a saved actor's configuration fingerprint must not change with this resume mechanic
    assert 'rng_continuation' not in train_mimo.acting_configuration(args)


def test_released_run_acted_without_a_frozen_actor_so_its_companion_holds_no_actor_rng():
    """The plan's open question (section 10: whether the released companion also carries an acting snapshot with
    a CUDA random state). The chain in train_mimo.py: `actor = None` at its start; an actor is built only under
    `if args.self_choice in ('on','shadow'):` (actor_for_run); the companion is written with
    `acting_snapshot=None if actor is None else actor.state_dict()`. So a run whose args.json records
    self_choice off saved no actor and no actor random state. Section 10 expected the test to read the companion
    on cpu; it reads args.json instead (the 372-byte sidecar episode_state.json beside the companion carries no
    key list, and torch.load would read the whole 621,243,950-byte companion, MANIFEST.json:100), and skips
    where the weights folder is absent."""
    path = RELEASED_RUN / 'args.json'
    if not path.exists():
        pytest.skip(f'{path} is not on this machine')
    args = json.loads(path.read_text(encoding='utf-8'))
    assert args['self_choice'] == 'off'
