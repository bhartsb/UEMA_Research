# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Bounded-memory caregiver recordings; append-only chunks and snapshot prefixes.

Files stay below FAT32's file-size limit. A restored simulation writes a new
chunk: the abandoned future remains recorded but never enters the restored tape.
Recording failure is reported locally and does not stop the infant's training.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import shutil
import time
import uuid

log = logging.getLogger(__name__)

# A REPLACE THAT A READER BLOCKS IS RETRIED, NOT FAILED. Windows refuses to replace a file that
# another process holds open. At 11:46:25.084 on 2026-09-22 the checkpoint archiver's tape pass
# was reading this recorder's manifest, the replace failed with "[WinError 5] Access is denied",
# and the recorder then dropped every event until its next 30 s volume check - 143 events - so
# that tape can never serve as a replay control. Measured the same day on this machine: a plain
# reader holding the file open blocks os.replace, and so does a reader opened with delete
# sharing, so the fix belongs here. The archiver holds a manifest for one read_text call, and
# that whole pass had reached its first move row 167 ms after the failure; these waits total
# 1.275 s, 7.6 times those 167 ms, and cost nothing when the first replace succeeds.
_REPLACE_WAITS_S = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)


def _json_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    for wait in _REPLACE_WAITS_S:
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(wait)
    temp.replace(path)                 # the last attempt raises, and the caller records the gap


class EventTape:
    def __init__(self, root, status_path, *, chunk_bytes=64 * 1024**2,
                 reserve_bytes=2 * 1024**3, check_seconds=30., flush_bytes=65536):
        if not 0 < chunk_bytes < 2**32 or flush_bytes <= 0:
            raise ValueError('invalid recording chunk/buffer size')
        self.root = Path(root)
        self.status_path = Path(status_path)
        self.chunk_bytes, self.reserve_bytes = chunk_bytes, reserve_bytes
        self.check_seconds, self.flush_bytes = check_seconds, flush_bytes
        self.tape_id = uuid.uuid4().hex
        self.spans = []
        self.count = 0
        self.dropped = 0
        self.error = None
        self.free_bytes = None
        self._buffer = bytearray()
        self._pending = 0
        self._last_check = float('-inf')
        self._chunk = None
        self._closed = False

    @property
    def manifest(self):
        return self.root / (self.tape_id + '.tape.json')

    def _metadata(self):
        return dict(format='caregiver_tape_v1', count=self.count,
                    dropped=self.dropped, spans=self.spans)

    def _status(self):
        status = dict(timestamp=time.time(), pid=os.getpid(),
                   recording=str(self.manifest), free_bytes=self.free_bytes,
                   reserve_bytes=self.reserve_bytes, dropped=self.dropped,
                   error=self.error, status='closed' if self._closed else ('recording' if self.error is None else 'recording_gap'))
        try:
            _json_file(self.status_path, status)
        except OSError as exc:
            log.error('Cannot write local caregiver storage status: %s', exc)

    def _failure(self, error):
        if self.error is None:
            log.error('Caregiver recording interrupted: %s', error)
        self.error = str(error)
        self.dropped += self._pending
        self._pending = 0
        self._buffer.clear()
        # A partial write is outside the manifest's committed byte range.
        self._chunk = None
        self._status()

    def _check(self):
        now = time.monotonic()
        if now - self._last_check < self.check_seconds:
            return self.error is None
        self._last_check = now
        try:
            # Never create a replacement directory for an absent removable drive.
            anchor = Path(self.root.anchor)
            if not anchor.exists():
                raise OSError(f'recording volume unavailable: {anchor}')
            self.free_bytes = shutil.disk_usage(anchor).free
            if self.free_bytes < self.reserve_bytes + self.flush_bytes:
                raise OSError(f'recording volume has {self.free_bytes} bytes free; reserve {self.reserve_bytes}')
            self.root.mkdir(parents=True, exist_ok=True)
            self.error = None
        except OSError as exc:
            self._failure(exc)
        self._status()
        return self.error is None

    def append(self, event):
        if self._closed:
            raise RuntimeError('closed caregiver tape')
        raw = (json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        if len(raw) > self.chunk_bytes:
            raise ValueError('one event exceeds recording chunk limit')
        if not self._check():
            self.dropped += 1
            return
        if self._chunk is not None and self.spans[-1]['bytes'] + len(self._buffer) + len(raw) > self.chunk_bytes:
            self.flush()
            self._chunk = None
        if self._chunk is None:
            self._chunk = uuid.uuid4().hex + '.jsonl'
            self.spans.append(dict(file=self._chunk, bytes=0, count=0))
        self._buffer.extend(raw)
        self._pending += 1
        if len(self._buffer) >= self.flush_bytes:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        try:
            with (self.root / self._chunk).open('ab') as stream:
                stream.write(self._buffer)
                stream.flush()
            self.spans[-1]['bytes'] += len(self._buffer)
            self.spans[-1]['count'] += self._pending
            self.count += self._pending
            self._buffer.clear()
            self._pending = 0
            _json_file(self.manifest, self._metadata())
        except OSError as exc:
            self._failure(exc)

    def __len__(self):
        return self.count + self._pending

    def __deepcopy__(self, memo):
        self.flush()
        # Neither view may append to a shared prefix after this point.
        self._chunk = None
        other = EventTape(self.root, self.status_path, chunk_bytes=self.chunk_bytes,
                          reserve_bytes=self.reserve_bytes, check_seconds=self.check_seconds,
                          flush_bytes=self.flush_bytes)
        other.spans = copy.deepcopy(self.spans, memo)
        other.count, other.dropped = self.count, self.dropped
        memo[id(self)] = other
        return other

    def __iter__(self):
        self.flush()
        if self.dropped:
            raise ValueError('recording has gaps and cannot serve as a replay control')
        yield from _read_spans(self.root, self.spans)

    def export_json(self, path):
        self.flush()
        if self.dropped:
            raise ValueError('recording has gaps and cannot serve as a replay control')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as out:
            out.write('[')
            for i, event in enumerate(self):
                if i:
                    out.write(',\n')
                out.write(json.dumps(event, allow_nan=False))
            out.write(']\n')

    def close(self):
        self.flush()
        if self.root.exists():
            try:
                _json_file(self.manifest, self._metadata())
            except OSError as exc:
                self._failure(exc)
        self._closed = True
        self._status()


# Where tools/archive_checkpoints.py moves a lineage's cold chunk files, one folder per
# recording folder: <TAPE_ARCHIVE>/<recording folder name>/<chunk>. A tape restored from a
# checkpoint names its chunks under the recording folder it was written in, so a chunk that
# has since moved is looked for here.
TAPE_ARCHIVE = Path('D:/UEMA_Research/external/D/UEMA_recordings/H84_checkpoint_archive/caregiver_tapes')


def chunk_path(root, name):
    root = Path(root)
    here = root / name
    if here.exists():
        return here
    moved = TAPE_ARCHIVE / root.name / name
    if moved.exists():
        return moved
    # moved some other way with a manifest row (a recording folder offloaded whole by
    # tools/offload_storage.py): found and hash-checked there; unknown -> `here`, as before
    from .storage_locations import locate
    return locate(here)


def _read_spans(root, spans):
    for span in spans:
        remaining = span['bytes']
        if not remaining:
            continue
        with chunk_path(root, span['file']).open('rb') as stream:
            for _ in range(span['count']):
                line = stream.readline(remaining)
                if not line.endswith(b'\n'):
                    raise ValueError('incomplete caregiver recording')
                remaining -= len(line)
                yield json.loads(line)
            if remaining:
                raise ValueError('caregiver recording count/bytes disagree')


class TapeReplay:
    """Sequential replay with one cached event; random seeks restart the reader."""
    def __init__(self, path):
        path = chunk_path(Path(path).parent, Path(path).name)   # a moved manifest too
        self.root = path.parent
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('format') != 'caregiver_tape_v1' or meta.get('dropped'):
            raise ValueError('invalid or incomplete caregiver recording')
        self.spans, self.count = meta['spans'], meta['count']
        # Validate the whole sequence without retaining it in memory.
        previous, n = float('-inf'), 0
        for event in _read_spans(self.root, self.spans):
            current = float(event['time'])
            if not previous <= current < float('inf'):
                raise ValueError('caregiver times must be finite and ordered')
            previous, n = current, n + 1
        if n != self.count or not n:
            raise ValueError('empty or inconsistent caregiver recording')
        self._index, self._event, self._iterator = -1, None, iter(_read_spans(self.root, self.spans))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index < self._index:
            self._index, self._iterator = -1, iter(_read_spans(self.root, self.spans))
        while self._index < index:
            self._event = next(self._iterator)
            self._index += 1
        return self._event
