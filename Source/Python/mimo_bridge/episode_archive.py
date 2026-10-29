# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The episode companion written one part at a time (Oct 1, 2026; docs/gain_of_function_v4/agents/save_memory_2026-10-01.md).

The companion stays what it has always been to every reader: torch.save's zip archive inside gzip, which
`torch.load(..., weights_only=False)` turns into dict(version, worlds, learner, trainer, config) with the same types and
values as before. Only the way it is written changed, because the old way held the whole episode in the trainer at
once at the moment Windows had least memory to give (the Oct 1 6:15 AM bugcheck came while that write was running):

  * each world arrives from its worker as one pickle (protocol 4), one world at a time, and is written straight into
    the archive as a byte record; data.pkl holds only `pickle.loads(<that record>)` for it, so torch.load rebuilds the
    world exactly as pickle would. The trainer never unpickles a world while saving;
  * the learner is pickled from the live model's tensors and arrays (no clone first): every tensor storage and every
    large numpy array is written into the archive the moment the pickler meets it, so data.pkl holds only the small
    Python objects around them. A large array is rebuilt on load as `torch.Tensor.numpy(<tensor over its record>)`,
    an ndarray with the saved dtype, shape and bytes. Two differences from torch.save, neither visible to the
    trainer's restore (which copies everything): a CUDA storage is recorded as a CPU one (the old path saved CPU
    copies), and a storage or array referenced twice is written once and loads shared (the old clones were
    separate).

Readers: continuation_state.load (and every tool that calls it) is unchanged; so is anything that opens the companion
with torch.load. The receipt (episode_state.json: sha256, bytes, step, storage_path, times) is written as before.
"""
import collections
import io
import pickle
import sys

import numpy as np

WORLD_PROTOCOL = 4          # a world's own pickle: numpy's protocol-2 reduce (byte order, layout) without the latin-1 detour
ARCHIVE_PROTOCOL = 2        # data.pkl, as torch.save writes it
MIN_ARRAY_BYTES = 1 << 16   # smaller arrays stay inside data.pkl, as torch.save would keep them

# numpy dtypes whose bytes torch can hold as a typed storage and give back through Tensor.numpy() unchanged
_ARRAY_DTYPES = {np.dtype(n): n for n in ('bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
                                          'float16', 'float32', 'float64', 'complex64', 'complex128')}


def world_blob(world):
    """One world's pickle, as the worker sends it and the archive stores it (pickle.loads gives the world back)."""
    return pickle.dumps(world, protocol=WORLD_PROTOCOL)


def _eligible(arr):
    return (type(arr) is np.ndarray and arr.nbytes >= MIN_ARRAY_BYTES and arr.dtype in _ARRAY_DTYPES
            and arr.dtype.isnative and arr.flags.c_contiguous)


class _Record:
    """A byte range already written into the archive, standing in for an ndarray (or a world's pickle bytes)."""
    __slots__ = ('key', 'dtype', 'shape', 'loads')

    def __init__(self, key, dtype, shape, loads=False):
        self.key, self.dtype, self.shape, self.loads = key, dtype, tuple(int(s) for s in shape), loads


class _StorageRef:
    """The persistent id of a written record, in the form torch.load's persistent_load reads."""
    __slots__ = ('key', 'storage_type', 'numel')

    def __init__(self, key, storage_type, numel):
        self.key, self.storage_type, self.numel = key, storage_type, numel


class _TensorRef:
    """Pickles as torch.save pickles a C-contiguous CPU tensor over a whole record."""
    __slots__ = ('storage', 'shape')

    def __init__(self, storage, shape):
        self.storage, self.shape = storage, shape

    def __reduce__(self):
        import torch
        stride, acc = [], 1
        for s in reversed(self.shape):
            stride.append(acc)
            acc *= max(int(s), 1)
        return (torch._utils._rebuild_tensor_v2,
                (self.storage, 0, self.shape, tuple(reversed(stride)), False, collections.OrderedDict()))


class EpisodeArchive:
    """torch's archive writer, fed part by part (records may be written in any order; data.pkl comes last)."""

    def __init__(self, fileobj):
        import torch
        self._torch = torch
        self.fileobj = fileobj
        self._sink = _Sink(fileobj)
        self.zip = torch._C.PyTorchFileWriter(self._sink, torch.serialization.get_crc32_options())
        # first, so a reader finds it before the records instead of seeking back to it at the end (torch.load
        # reads it before data.pkl; any record order is valid)
        self.zip.write_record('byteorder', sys.byteorder, len(sys.byteorder))
        self._next = 0
        self._storages = {}          # untyped storage _cdata -> key (one record per storage, as torch.save)
        self._storage_dtypes = {}    # data_ptr -> dtype (torch.save's mixed-view refusal)
        self._storage_types = {}
        self.record_bytes = 0
        self.records = 0
        self.data_pkl_bytes = None

    def _key(self):
        k = str(self._next)
        self._next += 1
        return k

    def _storage_type(self, dtype):
        name = self._storage_types.get(dtype)
        if name is None:
            torch = self._torch
            name = torch.TypedStorage(dtype=dtype, device='cpu', _internal=True)._pickle_storage_type()
            self._storage_types[dtype] = name
        return getattr(self._torch, name)

    def _write(self, key, data, nbytes):
        self.zip.write_record('data/' + key, data, nbytes)
        self.record_bytes += nbytes
        self.records += 1

    def _bytes_storage(self, data):
        """A torch storage over the bytes object's own memory (the writer's bytes overload would copy it first)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)        # read-only memory, only read here
            return self._torch.frombuffer(data, dtype=self._torch.uint8).untyped_storage()

    def put_world(self, blob):
        """A world's pickle bytes, written now; returns what data.pkl holds for that world."""
        key = self._key()
        self._write(key, self._bytes_storage(blob), len(blob))
        return _Record(key, 'uint8', (len(blob),), loads=True)

    def put_array(self, arr):
        key = self._key()
        import warnings
        with warnings.catch_warnings():
            # a read-only array is only read here; torch's "not writable" warning does not apply
            warnings.simplefilter('ignore', UserWarning)
            storage = self._torch.from_numpy(arr).untyped_storage()     # the array's own memory, not a copy
        if storage.nbytes() != arr.nbytes:
            raise ValueError('array memory is not one contiguous block of its own size')
        self._write(key, storage, arr.nbytes)
        return _Record(key, arr.dtype.name, arr.shape)

    def storage_id(self, obj):
        """torch.save's persistent_id for a storage, writing the storage the first time it is met."""
        torch = self._torch
        if isinstance(obj, torch.storage.TypedStorage):
            storage = obj._untyped_storage
            dtype = obj.dtype
            storage_type = getattr(torch, obj._pickle_storage_type())
            numel = obj._size()
        else:
            storage = obj
            dtype = torch.uint8
            storage_type = torch.UntypedStorage
            numel = storage.nbytes()
        if storage.data_ptr() != 0:
            seen = self._storage_dtypes.setdefault(storage.data_ptr(), dtype)
            if seen != dtype:
                raise RuntimeError('Cannot save multiple tensors or storages that view the same data as different types')
        key = self._storages.get(storage._cdata)
        if key is None:
            key = self._key()
            self._storages[storage._cdata] = key
            nbytes = storage.nbytes()
            # a CUDA storage is copied to the CPU for its own write and released after it (torch.save does the
            # same); it is recorded as a CPU storage, as the trainer's old CPU copies were
            cpu = storage.cpu() if storage.device.type != 'cpu' else storage
            self._write(key, cpu, nbytes)
            del cpu
        return ('storage', storage_type, key, 'cpu', numel)

    def finish(self, obj):
        """Pickle `obj` (worlds already written, everything else written as met), then close the archive."""
        buf = io.BytesIO()
        _ArchivePickler(buf, self).dump(obj)
        data = buf.getvalue()           # the buffer itself, not a copy (no view of it is open)
        buf.close()
        self.data_pkl_bytes = len(data)
        self.zip.write_record('data.pkl', self._bytes_storage(data) if data else data, len(data))
        del data
        self.zip.write_end_of_file()
        self.fileobj.flush()

    def abandon(self):
        """After a failed save: close torch's writer without touching the file. Left to its destructor, the writer
        would write its end records into the closed file and C++ would abort the whole trainer (measured in this
        branch's failed-world test: 'Fatal Python error: Aborted')."""
        self._sink.dead = True
        try:
            self.zip.write_end_of_file()
        except Exception:
            pass


class _Sink:
    """The file torch's writer writes into; once dead, writes are dropped (see EpisodeArchive.abandon)."""

    def __init__(self, f):
        self.f, self.dead = f, False

    def write(self, data):
        if self.dead:
            return len(data)
        return self.f.write(data)

    def flush(self):
        if not self.dead:
            self.f.flush()


class _ArchivePickler(pickle.Pickler):
    def __init__(self, f, archive):
        super().__init__(f, protocol=ARCHIVE_PROTOCOL)
        self.archive = archive

    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ('storage', obj.storage_type, obj.key, 'cpu', obj.numel)
        torch = self.archive._torch
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            return self.archive.storage_id(obj)
        return None

    def reducer_override(self, obj):
        if type(obj) is _Record:
            return self._record_reduce(obj)
        if type(obj) is np.ndarray and _eligible(obj):
            return self._record_reduce(self.archive.put_array(obj))
        return NotImplemented

    def _record_reduce(self, rec):
        torch = self.archive._torch
        dtype = np.dtype(rec.dtype)
        tdtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        numel = 1
        for s in rec.shape:
            numel *= s
        tensor = _TensorRef(_StorageRef(rec.key, self.archive._storage_type(tdtype), numel), rec.shape)
        array = (torch.Tensor.numpy, (tensor,))
        if rec.loads:
            # the world: pickle.loads over the record's bytes gives the object the worker pickled
            return (pickle.loads, (_Reduced(array),))
        return array


class _Reduced:
    """An object that pickles as a given reduce tuple (lets a reduce be nested inside another's arguments)."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return self.value
