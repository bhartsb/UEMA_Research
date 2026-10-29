# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import json
from types import SimpleNamespace

import pytest

from mimo_bridge.event_tape import EventTape, TapeReplay


def test_snapshot_branch_excludes_abandoned_future_and_replays(tmp_path):
    tape = EventTape(tmp_path/'usb', tmp_path/'status.json', reserve_bytes=0, chunk_bytes=128)
    tape.append(dict(time=0, phrase='look'))
    snapshot = copy.deepcopy(tape)
    tape.append(dict(time=1, phrase='abandoned'))
    tape.close()
    restored = copy.deepcopy(snapshot)
    restored.append(dict(time=1, phrase='touch'))
    restored.close()
    replay = TapeReplay(restored.manifest)
    assert len(replay) == 2
    assert replay[1]['phrase'] == 'touch'
    assert replay[0]['phrase'] == 'look'
    assert replay[1]['phrase'] == 'touch'
    assert TapeReplay(tape.manifest)[1]['phrase'] == 'abandoned'


def test_many_events_stay_bounded_and_chunks_fit(tmp_path):
    tape = EventTape(tmp_path/'usb', tmp_path/'status.json', reserve_bytes=0,
                     chunk_bytes=1024, flush_bytes=256)
    for i in range(2000):
        tape.append(dict(time=i, target_actual=[1,2,3]))
        assert len(tape._buffer) < 256
    tape.close()
    assert max(p.stat().st_size for p in tape.root.glob('*.jsonl')) <= 1024
    assert len(TapeReplay(tape.manifest)) == 2000
    tape.export_json(tmp_path/'export.json')
    assert len(json.loads((tmp_path/'export.json').read_text())) == 2000


def test_low_space_reports_gap_and_rejects_replay_without_stopping(tmp_path, monkeypatch):
    monkeypatch.setattr('mimo_bridge.event_tape.shutil.disk_usage', lambda _: SimpleNamespace(free=1))
    tape = EventTape(tmp_path/'usb', tmp_path/'status.json', check_seconds=0)
    tape.append(dict(time=0))
    monkeypatch.setattr('mimo_bridge.event_tape.shutil.disk_usage', lambda _: SimpleNamespace(free=10*1024**3))
    tape.append(dict(time=1))
    tape.close()
    assert tape.dropped == 1
    assert json.loads((tmp_path/'status.json').read_text())['dropped'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        TapeReplay(tape.manifest)


def _collide(tape, hold_s):
    """Hold the tape's manifest open from another thread for hold_s while the recorder flushes
    one event into it. Returns the seconds the flush took."""
    import threading
    import time
    opened, done = threading.Event(), threading.Event()

    def reader():
        with open(tape.manifest, 'rb'):
            opened.set()
            time.sleep(hold_s)
        done.set()
    t = threading.Thread(target=reader)
    t.start()
    opened.wait(5)
    start = time.perf_counter()
    tape.append(dict(time=1))                      # flush_bytes=1: this append replaces the manifest
    took = time.perf_counter() - start
    t.join(5)
    return took


def test_a_reader_holding_the_manifest_open_delays_the_recorder_and_drops_nothing(tmp_path):
    """2026-09-22 11:46:25: the archiver's tape pass held a manifest open while the trainer's
    recorder replaced it; the replace failed with WinError 5 and 143 events were dropped."""
    import os
    tape = EventTape(tmp_path/'usb', tmp_path/'status.json', reserve_bytes=0, flush_bytes=1)
    tape.append(dict(time=0))                      # the manifest exists
    took = _collide(tape, hold_s=0.15)
    assert tape.error is None and tape.dropped == 0
    if os.name == 'nt':
        assert took >= 0.1, 'no collision happened, so this test proved nothing'
    tape.close()
    assert [e['time'] for e in TapeReplay(tape.manifest)] == [0, 1]


def test_without_the_retries_the_same_collision_drops_events(tmp_path, monkeypatch):
    """The control: the collision above is real on this platform, not an artefact of the test."""
    import os
    if os.name != 'nt':
        pytest.skip('only Windows refuses to replace a file another process holds open')
    monkeypatch.setattr('mimo_bridge.event_tape._REPLACE_WAITS_S', ())
    tape = EventTape(tmp_path/'usb', tmp_path/'status.json', reserve_bytes=0, flush_bytes=1)
    tape.append(dict(time=0))
    _collide(tape, hold_s=0.15)
    assert tape.error is not None and ('Access is denied' in tape.error or 'WinError 5' in tape.error)
    tape.append(dict(time=2))                      # inside the 30 s check interval: dropped
    assert tape.dropped >= 1


def test_disconnected_volume_reports_local_error(tmp_path, monkeypatch):
    def removed(_):
        raise OSError('volume disconnected')
    monkeypatch.setattr('mimo_bridge.event_tape.shutil.disk_usage', removed)
    tape = EventTape(tmp_path/'usb', tmp_path/'status.json')
    tape.append(dict(time=0))
    tape.close()
    assert 'disconnected' in json.loads((tmp_path/'status.json').read_text())['error']
