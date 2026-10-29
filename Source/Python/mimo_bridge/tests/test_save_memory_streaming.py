# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The episode companion written one world at a time loads exactly as the whole-state torch.save did (2026-10-01).

docs/gain_of_function_v4/agents/save_memory_2026-10-01.md: the save that was running when Windows stopped at
6:15 AM on Oct 1 held all eight worlds, a clone of the learner and a walk over 87 million objects at once. The new
writer (mimo_bridge/episode_archive.py, continuation_state.save) streams; these tests hold it to the old file's
meaning: every reader that calls torch.load or continuation_state.load gets the same types, shapes, dtypes and
values, the receipt is the same, a failed write leaves nothing behind, and the pool asks one worker at a time.
No world, model or trainer is built: the sources and the learner are small stand-ins.
"""
import collections
import dataclasses
import gzip
import json
import multiprocessing as mp
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge import continuation_state as CS
from mimo_bridge import datasource as DS
from mimo_bridge.episode_archive import EpisodeArchive, MIN_ARRAY_BYTES, world_blob


@dataclasses.dataclass
class Held:
    order: np.ndarray
    note: str


class Plain:
    """A plain object in a world (the caregiver, the skin and the poolers are such objects)."""

    def __init__(self, n):
        self.history = [float(i) / 7 for i in range(n)]
        self.big = np.arange(n * 40, dtype=np.float64)


def world(i, events=2000):
    rng = np.random.default_rng(i)
    return dict(
        env_kwargs=dict(seed=i, scene='crib', tag='_w%d' % i),
        world=dict(
            caregiver=dict(events=[dict(time=0.05 * k, target_actual=[0.1 * k, 0.2, 0.3], visible=bool(k % 2),
                                        scene_phase='engaged') for k in range(events)]),
            attrs=dict(skin=Plain(300), rng=rng.bit_generator.state),
            integration_state=rng.normal(size=5000),
            fortran=np.asfortranarray(rng.normal(size=(64, 300))),
            big_endian=rng.normal(size=9000).astype('>f8'),
            small=np.arange(5, dtype=np.int16),
            flags=rng.random(70000) > 0.5,
            half=rng.normal(size=40000).astype(np.float16),
            objects=np.array([{'a': 1}, None], dtype=object),
            scalar=np.float32(2.5)),
        echo=rng.normal(size=(26, 9)).astype(np.float32),
        payload=dict(sim_time=12.5 + i, patches=rng.normal(size=(300, 4)).astype(np.float32), frame_bytes=b'\x00\x01' * 5000),
        elapsed=3.25 + i)


def learner_model():
    base = torch.arange(24000, dtype=torch.float32).reshape(40, 600)
    replay = SimpleNamespace(tokens=torch.randn(2, 64, 9, 192).half(), ptr=7, count=64, episode=torch.zeros(2, 64, dtype=torch.int32),
                             device=torch.device('cpu'))
    loss = SimpleNamespace(_vicreg_history={'vision': collections.deque([torch.randn(8, 4) for _ in range(3)], maxlen=5)})
    gen = torch.Generator()
    gen.manual_seed(3)
    model = SimpleNamespace(
        _ctx=base, _view=base[3:9], _transposed=base.t(), _shared=base.view(600, 40), last_order=np.ones((26, 9), np.float32),
        last_big=np.linspace(0, 1, 50000), last_fortran=np.asfortranarray(np.arange(90000.).reshape(300, 300)),
        context_buffer=torch.randn(2, 8, 192), _held=Held(np.zeros(30000), 'x'),
        _ring=collections.deque([np.full(20000, k, np.float32) for k in range(3)], maxlen=4), _none=None,
        replay=replay, movement=None, curiosity=None, consequence_lesson=None,
        memory_history=None, self_history=None, self_rehearsal=None, loss_manager=loss,
        placeholders=SimpleNamespace(_generators={}), noise_rng=gen, device=torch.device('cpu'))
    return model


class ListSource:
    """A source with only save_full_state (the fallback continuation_state.world_blobs pickles)."""

    def __init__(self, worlds):
        self.worlds = worlds

    def save_full_state(self):
        return [w for w in self.worlds]


def same(a, b, where='state'):
    """Exact equality of what a reader gets: types, keys in order, dtypes, shapes, layout and values."""
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b), where
        for k in a:
            same(a[k], b[k], '%s.%s' % (where, k))
    elif isinstance(a, collections.deque):
        assert a.maxlen == b.maxlen and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, '%s[%d]' % (where, i))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, '%s[%d]' % (where, i))
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert a.flags.c_contiguous == b.flags.c_contiguous and a.flags.f_contiguous == b.flags.f_contiguous, where
        if a.dtype == object:
            assert a.tolist() == b.tolist(), where
        else:
            assert np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'), where
    elif torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride(), where
        assert a.device == b.device and torch.equal(a, b), where
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            same(getattr(a, f.name), getattr(b, f.name), '%s.%s' % (where, f.name))
    elif isinstance(a, Plain):
        same(vars(a), vars(b), where)
    elif isinstance(a, torch.Generator):
        assert torch.equal(a.get_state(), b.get_state()), where
    else:
        assert a == b, (where, a, b)


def old_state(model, worlds, trainer):
    """What continuation_state.save built before 2026-10-01."""
    return dict(version=CS.VERSION, worlds=[dict(w) for w in worlds], learner=CS.capture_learner(model),
                trainer=CS.tree(trainer), config=dict(example='test'))


def torch_load(path):
    with gzip.open(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=False)


def test_streamed_companion_loads_exactly_as_the_whole_state_torch_save(tmp_path):
    torch.manual_seed(0)
    model = learner_model()
    worlds = [world(0), world(1)]
    trainer = dict(step=300, held_motor=np.zeros((2, 26, 9), np.float32), bout_rng={'state': 5})
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()
    # the RNG states captured with the learner must be the same in both
    rng_state = torch.get_rng_state()
    old_receipt = CS.write_episode(old_dir, old_state(model, worlds, trainer))
    torch.set_rng_state(rng_state)
    new_receipt = CS.save(new_dir, ListSource(worlds), model, trainer, dict(example='test'))
    a, b = torch_load(old_dir / 'episode_state.pt.gz'), torch_load(new_dir / 'episode_state.pt.gz')
    same(a, b)
    same(CS.load(old_dir), CS.load(new_dir))
    for k in ('version', 'step', 'times'):
        assert old_receipt[k] == new_receipt[k]
    assert new_receipt['times'] == [12.5, 13.5]
    assert not list(new_dir.glob('*.tmp'))


def test_large_arrays_and_tensors_become_records_not_pickle_text(tmp_path):
    model = learner_model()
    state = dict(learner=CS.capture_learner(model, copy_state=False))
    path = tmp_path / 'a.pt.gz'
    with gzip.open(path, 'wb', compresslevel=1) as f:
        archive = EpisodeArchive(f)
        archive.finish(state)
    # 3 deque arrays + last_big + _held.order + the dense tensors: data.pkl keeps only the small objects
    assert archive.data_pkl_bytes < 64 * 1024
    assert archive.record_bytes > 1_000_000
    same(state['learner']['data']['_ring'], torch_load(path)['learner']['data']['_ring'])


def test_tree_ref_passes_dense_tensors_and_arrays_on_and_copies_views():
    model = learner_model()
    got = CS.capture_learner(model, copy_state=False)['data']
    assert got['_ctx'].data_ptr() == model._ctx.data_ptr()
    assert got['last_big'] is model.last_big
    assert got['_view'].data_ptr() != model._view.data_ptr() and torch.equal(got['_view'], model._view)
    assert got['_transposed'].data_ptr() != model._transposed.data_ptr()
    assert got['_transposed'].stride() == CS.tree(model._transposed).stride()
    with pytest.raises(TypeError, match='explicit serializer'):
        CS.tree_ref(object())


def test_a_failed_world_leaves_no_temporary_file_and_no_receipt(tmp_path):
    class Broken:
        def iter_world_blobs(self):
            blob = world_blob(world(0))
            yield dict(sim_time=1.0, nbytes=len(blob)), blob
            raise RuntimeError('worker 1 snapshot failed')

    with pytest.raises(RuntimeError, match='worker 1'):
        CS.save(tmp_path, Broken(), learner_model(), dict(step=3), {})
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_storage_root_receipt_names_the_companion_where_it_is(tmp_path):
    ckpt, root = tmp_path / '.tmp_ckpt_step_5_x', tmp_path / 'caregiver' / 'checkpoint_state' / 'run'
    ckpt.mkdir()
    receipt = CS.save(ckpt, ListSource([world(0)]), learner_model(), dict(step=5), {}, storage_root=str(root),
                      published_path=str(tmp_path / 'ckpt_step_5'))
    stored = Path(receipt['storage_path'])
    assert stored.parent == root.resolve() and stored.name.startswith('ckpt_step_5_') and stored.exists()
    assert receipt['bytes'] == stored.stat().st_size and receipt['sha256'] == CS.file_sha256(stored)
    assert json.loads((ckpt / 'episode_state.json').read_text(encoding='utf-8')) == receipt
    assert CS.load(ckpt)['trainer']['step'] == 5


class FakeEnv:
    def __init__(self, i, log):
        self.i, self.log = i, log

    def save_full_state(self):
        self.log.append('snapshot %d' % self.i)
        return world(self.i, events=300)['world']


def test_the_pool_asks_one_worker_at_a_time_and_the_bytes_are_the_world(tmp_path):
    ctx = mp.get_context('spawn')
    log, threads, pool = [], [], object.__new__(DS.PoolSource)
    pool.conns, pool.n_envs = [], 3
    for i in range(3):
        parent, child = ctx.Pipe()
        w = world(i, events=300)

        def serve(conn=child, i=i, w=w):
            cmd, _ = conn.recv()
            assert cmd == 'save_full_state_blob'
            DS._send_world_blob(conn, FakeEnv(i, log), w['env_kwargs'], w['echo'], w['payload'], 0.0)

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        threads.append(t)
        pool.conns.append(parent)
    got = []
    for header, blob in pool.iter_world_blobs():
        log.append('received %d' % len(got))
        import pickle
        got.append(pickle.loads(blob))
        assert header['nbytes'] == len(blob) and header['sim_time'] == 12.5 + len(got) - 1
        del blob
    for t in threads:
        t.join(5)
    assert log == ['snapshot 0', 'received 0', 'snapshot 1', 'received 1', 'snapshot 2', 'received 2']
    import pickle
    for i, g in enumerate(got):
        # what the old 'save_full_state' answer gave the trainer: the world through the pipe's pickle (protocol 4,
        # which, like torch.save's protocol 2, gives a non-native array back in native byte order)
        expected = pickle.loads(pickle.dumps(world(i, events=300), protocol=4))
        same(expected['world'], g['world'])
        same(expected['payload'], g['payload'])
        assert g['env_kwargs'] == expected['env_kwargs'] and list(g) == ['env_kwargs', 'world', 'echo', 'payload', 'elapsed']


def test_require_save_bytes_refuses_a_part_larger_than_the_free_space(tmp_path):
    from lem.training.checkpoint import require_save_bytes, require_save_space
    assert require_save_bytes(str(tmp_path), 1000)['estimated_required_bytes'] == int(1000 * 1.25) + 64 * 1024 ** 2
    with pytest.raises(OSError, match='needs approximately'):
        require_save_bytes(str(tmp_path), 10 ** 18)
    est = require_save_space(str(tmp_path), dict(a=np.zeros(1000)))['estimated_required_bytes']
    assert est > int(8000 * 1.25) + 64 * 1024 ** 2


def test_threshold_keeps_small_arrays_in_the_pickle(tmp_path):
    state = dict(small=np.zeros(MIN_ARRAY_BYTES // 8 - 1), big=np.zeros(MIN_ARRAY_BYTES // 8))
    path = tmp_path / 's.pt.gz'
    with gzip.open(path, 'wb') as f:
        archive = EpisodeArchive(f)
        archive.finish(state)
    assert archive.records == 1
    same(state, torch_load(path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='the CUDA learner path needs a GPU (run in a training pause)')
def test_a_cuda_learner_saves_as_the_old_cpu_copies(tmp_path):
    """Production's learner lives on the GPU: tree_ref passes CUDA tensors on and the archive copies each storage to
    the CPU for its own write. The file must hold what the old tree() CPU clones held."""
    model = learner_model()
    dev = torch.device('cuda:0')
    base = torch.randn(40, 600, device=dev)
    model._cuda_dense = base
    model._cuda_view = base[3:9]
    model._cuda_t = base.t()
    model._cuda_bf16 = torch.randn(64, 64, device=dev).to(torch.bfloat16)
    model._cuda_shared = base.view(600, 40)
    model.context_buffer = model.context_buffer.to(dev)
    trainer = dict(step=11)
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()
    worlds = [world(0, events=50)]
    rng = torch.get_rng_state()
    CS.write_episode(old_dir, old_state(model, worlds, trainer))
    torch.set_rng_state(rng)
    CS.save(new_dir, ListSource(worlds), model, trainer, dict(example='test'))
    a, b = CS.load(old_dir), CS.load(new_dir)
    same(a, b)
    assert all(t.device.type == 'cpu' for k, t in b['learner']['data'].items() if torch.is_tensor(t))


def test_a_worker_whose_world_cannot_be_pickled_answers_and_keeps_serving():
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()

    class Unpicklable:
        def __reduce__(self):
            raise TypeError('no pickling')

    class BadEnv:
        def save_full_state(self):
            return dict(x=Unpicklable())

    w = world(0, events=10)
    t = threading.Thread(target=DS._send_world_blob, args=(child, BadEnv(), w['env_kwargs'], w['echo'], w['payload'], 0.0))
    t.start()
    tag, text = parent.recv()
    t.join(5)
    assert tag == 'err' and 'no pickling' in text and not parent.poll(0.2)
