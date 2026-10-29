# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The caregiver's event list is bounded and every reader reads what it read before (2026-10-02).

Record: docs/gain_of_function_v4/agents/caregiver_event_list_2026-10-02.md. No world is built: the
caregiver's event handling needs no MuJoCo model (Caregiver._event touches the env only for a
style or a phrase, which these events do not carry). CPU only.
"""
from __future__ import annotations

import ast
import copy
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from mimo_bridge import caregiver as C
from mimo_bridge.event_tape import EventTape
from mimo_bridge.tools.observer import baseline as B

ROOT = Path(C.__file__).resolve().parent              # Source/Python/mimo_bridge
LONG = 250_000                                        # past H84's 203,873 at step 455,800


def bare(mode='responsive', events=None):
    """A Caregiver without a world: the state the event code reads, nothing else."""
    c = C.Caregiver.__new__(C.Caregiver)
    c.mode = mode
    c.state = dict(time=0., events=[] if events is None else events, replies=0, style=0,
                   target_goal=np.zeros(3), visible=True)
    return c


def step(c, k):
    """Event k of the responsive caregiver, one per call: a pose record, and every 600th a goal
    event through _event instead (the code path speech and scene changes take)."""
    c.state['time'] = 0.01 * (k + 1)
    if k % 600 == 0:
        c._event(None, dict(target=[0.1, 0.2, 0.01 * (k % 7)], visible=bool(k % 2)))
    else:
        c._record(dict(time=c.state['time'], target_actual=[0.1, 0.2, 0.3], scene_phase='engaged'))


def readers(c):
    """What every reader of the event list reads, at one moment (file:line at 66c51897e)."""
    ev = c.state['events']
    snap = c.snapshot()                                         # caregiver.py:460, env.py:1898
    other = bare(c.mode)
    other.restore(snap)                                         # caregiver.py:462, env.py:1919
    world = dict(caregiver=snap)
    return dict(
        report_events=len(ev),                                  # env.py:3545
        report_recording=hasattr(ev, 'manifest'),               # env.py:3547-3548
        truth=bool(ev),
        snapshot_events=len(snap['events']),
        restored_events=len(other.state['events']),
        observer_count=len(B.detach_recording(world)['caregiver']['events']),   # baseline.py:157-161
        observer_row=len(world['caregiver']['events']),         # baseline.py:278
        other_state={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in c.state.items() if k != 'events'})


def run(monkeypatch, limit):
    monkeypatch.setattr(C, 'EVENT_LIST_LIMIT', limit)
    c, out = bare(), []
    for k in range(LONG):
        step(c, k)
        if k + 1 in (1, 19_999, 20_000, 20_001, 20_002, 100_000, 203_792, LONG):
            out.append(readers(c))
        if k + 1 in (60_000, 203_792):
            # a resume halfway: the world continues from its own restored copy
            snap = c.snapshot()
            c = bare()
            c.restore(snap)
    return c, out


def test_every_reader_reads_the_same_over_a_long_history(monkeypatch):
    """250,000 events with two resumes: the unbounded list (the code before) and the bounded one
    give every reader the same answer at every sampled moment."""
    before_c, before = run(monkeypatch, float('inf'))
    assert type(before_c.state['events']) is list and len(before_c.state['events']) == LONG
    del before_c
    after_c, after = run(monkeypatch, 20_000)
    assert before == after
    ev = after_c.state['events']
    assert isinstance(ev, C.EventLog) and len(ev) == LONG and len(ev.recent) == C.EVENT_LOG_WINDOW
    assert ev.recent[-1]['time'] == pytest.approx(0.01 * LONG)


def test_a_list_up_to_the_limit_is_the_plain_list_its_tools_export(tmp_path):
    """The preview and trial tools' runs (12,500 events at most) see the list they always saw."""
    c = bare()
    for k in range(C.EVENT_LIST_LIMIT):
        step(c, k)
    ev = c.state['events']
    assert type(ev) is list and len(ev) == C.EVENT_LIST_LIMIT
    c.export_events(tmp_path / 'events.json')
    assert (tmp_path / 'events.json').read_text(encoding='utf-8') == json.dumps(ev, indent=2, allow_nan=False)
    assert [e for e in c.state['events'] if 'target' in e] == [e for e in ev if 'target' in e]


def test_one_past_the_limit_keeps_the_newest_events_and_the_count():
    c = bare()
    full = []
    for k in range(C.EVENT_LIST_LIMIT + 1):
        step(c, k)
        if type(c.state['events']) is list:
            full = c.state['events']
    ev = c.state['events']
    assert isinstance(ev, C.EventLog)
    assert len(ev) == C.EVENT_LIST_LIMIT + 1 and ev.dropped == C.EVENT_LIST_LIMIT + 1 - C.EVENT_LOG_WINDOW
    assert list(ev.recent) == full[-C.EVENT_LOG_WINDOW:]
    for k in range(C.EVENT_LIST_LIMIT + 1, C.EVENT_LIST_LIMIT + 251):
        step(c, k)
    assert len(c.state['events']) == C.EVENT_LIST_LIMIT + 251
    assert c.state['events'].recent[-1]['time'] == pytest.approx(0.01 * (C.EVENT_LIST_LIMIT + 251))


def test_a_trimmed_history_refuses_to_pass_for_the_whole_sequence(tmp_path):
    c = bare(events=list(range(C.EVENT_LIST_LIMIT + 5)))
    c._record(dict(time=1.))
    with pytest.raises(ValueError, match='whole sequence'):
        c.export_events(tmp_path / 'events.json')
    with pytest.raises(ValueError, match='whole sequence'):
        [e for e in c.state['events']]
    with pytest.raises(TypeError):
        json.dumps(c.state['events'])
    assert not (tmp_path / 'events.json').exists()


def test_a_replay_caregiver_is_never_trimmed(tmp_path):
    """Its events are the replayed sequence, which caregiver_learning_trial exports (9,000
    updates = 45,000 events at its defaults)."""
    replay = [dict(time=0.01 * k, target_actual=[0., 0., 0.01 * k]) for k in range(C.EVENT_LIST_LIMIT + 500)]
    c = bare(mode='replay')
    for e in replay:
        c._event(None, e)
    assert type(c.state['events']) is list and c.state['events'] == replay
    other = bare(mode='replay')
    other.restore(c.snapshot())
    assert type(other.state['events']) is list and len(other.state['events']) == len(replay)
    c.export_events(tmp_path / 'adult.json')
    assert json.loads((tmp_path / 'adult.json').read_text(encoding='utf-8')) == replay


class Counted:
    """An event that counts how often it is deep-copied."""
    copies = 0

    def __init__(self, k):
        self.k = k

    def __deepcopy__(self, memo):
        Counted.copies += 1
        return Counted(self.k)

    def __eq__(self, other):
        return isinstance(other, Counted) and other.k == self.k


def test_restoring_an_old_full_list_copies_only_the_window_and_leaves_the_save_alone():
    """Every H84 world saved before this change holds the full list (203,873 at most)."""
    saved_list = [Counted(k) for k in range(203_873)]
    saved = dict(time=22790.5, events=saved_list, replies=7, style=1)
    Counted.copies = 0
    c = bare()
    c.restore(saved)
    assert Counted.copies == C.EVENT_LOG_WINDOW
    ev = c.state['events']
    assert isinstance(ev, C.EventLog) and len(ev) == 203_873
    assert [e.k for e in ev.recent] == list(range(203_873 - C.EVENT_LOG_WINDOW, 203_873))
    assert saved['events'] is saved_list and len(saved_list) == 203_873
    assert all(type(e) is Counted for e in saved_list[-3:]) and ev.recent[-1] is not saved_list[-1]
    again = bare()
    again.restore(saved)
    assert again.state['events'] == ev and again.state['events'] is not ev
    assert {k: v for k, v in again.state.items() if k != 'events'} == {k: v for k, v in saved.items() if k != 'events'}


def test_a_bounded_snapshot_copies_only_the_window_and_is_its_own_copy():
    c = bare(events=[Counted(k) for k in range(C.EVENT_LIST_LIMIT + 1)])
    c._record(Counted(-1))
    Counted.copies = 0
    snap = c.snapshot()
    assert Counted.copies == C.EVENT_LOG_WINDOW
    c._record(Counted(-2))
    assert len(snap['events']) == C.EVENT_LIST_LIMIT + 2 and len(c.state['events']) == C.EVENT_LIST_LIMIT + 3
    assert snap['events'].recent[-1] == Counted(-1)


def test_the_saved_world_pickles_small_and_loads_equal():
    """The companion stores each world as one protocol-4 pickle (episode_archive.world_blob)."""
    from mimo_bridge.episode_archive import world_blob
    event = dict(time=1., target_actual=[0.1, 0.2, 0.3], head_actual=[0.4, 0.5, 0.6], scene_phase='engaged')
    full = [dict(event, time=0.01 * k) for k in range(30_000)]
    c = bare()
    c.restore(dict(time=300., events=full, replies=3))
    world = dict(caregiver=c.snapshot())
    blob = world_blob(world)
    back = pickle.loads(blob)
    # saved as babble's counter (the review of 2026-10-02): the count, not the kept events
    assert isinstance(world['caregiver']['events'], C.EventLog)
    assert type(back['caregiver']['events']) is B.DiscardedEvents and len(back['caregiver']['events']) == 30_000
    assert {k: v for k, v in back['caregiver'].items() if k != 'events'} == \
        {k: v for k, v in world['caregiver'].items() if k != 'events'}
    assert len(blob) * 100 < len(world_blob(dict(caregiver=dict(time=300., events=full, replies=3))))


BASE = '66c51897e'      # production's code when this branch forked


def production_module(tmp_path, rel, name):
    """Production's file `rel` (under Source/Python/mimo_bridge) at BASE, copied to tmp_path and
    imported as the module `name` inside the mimo_bridge package (its relative imports resolve)."""
    import importlib.util
    import subprocess
    repo = ROOT.parents[2]
    try:
        source = subprocess.run(['git', 'show', f'{BASE}:Source/Python/mimo_bridge/{rel}'], cwd=repo,
                                capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f'production code at {BASE} not readable here: {e}')
    path = tmp_path / f'{BASE}_{rel.replace("/", "_")}'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_save_made_by_this_code_loads_in_production_code_and_back(tmp_path, monkeypatch):
    """Rollback, a branch forked before the merge, or production's code pinned for a comparison
    must load a world this code saved (review of 2026-10-02: an EventLog pickled under its own
    name raised AttributeError: Can't get attribute 'EventLog' in production's caregiver.py)."""
    import sys
    from mimo_bridge.episode_archive import world_blob
    # under their own names (not registered until swapped in below), so a pickle made there names
    # the modules a production process has
    old_cg = production_module(tmp_path, 'caregiver.py', 'mimo_bridge.caregiver')
    old_b = production_module(tmp_path, 'tools/observer/baseline.py', 'mimo_bridge.tools.observer.baseline')
    assert sys.modules['mimo_bridge.caregiver'] is C and old_cg is not C
    assert not hasattr(old_cg, 'EventLog') and hasattr(old_b, 'DiscardedEvents')
    total = C.EVENT_LIST_LIMIT + 1_234
    c = bare(events=[dict(time=0.01 * k, target_actual=[0.1, 0.2, 0.3]) for k in range(total - 1)])
    step(c, total - 1)
    assert isinstance(c.state['events'], C.EventLog) and len(c.state['events']) == total
    blob = world_blob(dict(caregiver=c.snapshot()))

    # production's code: its caregiver.py and baseline.py stand in for this branch's
    with monkeypatch.context() as m:
        m.setitem(sys.modules, 'mimo_bridge.caregiver', old_cg)
        m.setitem(sys.modules, 'mimo_bridge.tools.observer.baseline', old_b)
        world = pickle.loads(blob)
        ev = world['caregiver']['events']
        assert type(ev) is old_b.DiscardedEvents and len(ev) == total
        prod = old_cg.Caregiver.__new__(old_cg.Caregiver)
        prod.mode, prod.state = 'responsive', dict(events=[])
        prod.restore(world['caregiver'])
        prod.state['events'].append(dict(time=1.))                      # its pose append
        prod._event(None, dict(target=[0.1, 0.2, 0.3], visible=True))     # its goal append
        assert len(prod.state['events']) == total + 2
        resaved = world_blob(dict(caregiver=prod.snapshot()))

    # and back in this code, from both saves
    for saved, count in ((blob, total), (resaved, total + 2)):
        world = pickle.loads(saved)
        assert type(world['caregiver']['events']) is B.DiscardedEvents
        new = bare()
        new.restore(world['caregiver'])
        for k in range(count, count + 7):
            step(new, k)
        assert len(new.state['events']) == count + 7
        other = bare()
        other.restore(new.snapshot())
        assert len(other.state['events']) == count + 7
        with pytest.raises(ValueError, match='whole sequence'):
            new.export_events(tmp_path / 'events.json')
        assert not (tmp_path / 'events.json').exists()
        assert len(pickle.loads(world_blob(dict(caregiver=new.snapshot())))['caregiver']['events']) == count + 7


def test_an_in_process_copy_stays_an_event_log():
    log = C.EventLog([dict(time=float(k)) for k in range(5)], total=50_000)
    for copied in (copy.deepcopy(log), copy.copy(log)):
        assert type(copied) is C.EventLog and copied == log and copied is not log
        copied.append(dict(time=9.))
        assert len(copied) == 50_001 and len(log) == 50_000


def test_tapes_counters_and_logs_pass_through_unchanged(tmp_path):
    c = bare()
    tape = EventTape(tmp_path / 'rec', tmp_path / 'status.json', reserve_bytes=0)
    counter = B.DiscardedEvents(500_000)
    log = C.EventLog(range(5), total=500_000)
    for value in (tape, counter, log, list(range(C.EVENT_LIST_LIMIT))):
        assert c._bounded_events(value) is value
    tape.close()


def test_every_append_goes_through_the_bound():
    """No code in caregiver.py appends to the event list except Caregiver._record."""
    tree = ast.parse((ROOT / 'caregiver.py').read_text(encoding='utf-8'))
    appends = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'append' and 'events' in ast.unparse(node.func.value)):
                    appends.append((fn.name, ast.unparse(node.func.value)))
    assert {name for name, _ in appends} == {'_record'}, appends


# Every line outside the tests that reads a caregiver's event list, with what it reads. A new
# reader must be added here with what it needs, after checking it against the bound.
KNOWN_READERS = {
    'caregiver.py': 'the caregiver itself: _record, export_events, close_recording',
    'env.py': 'len() and hasattr(manifest) only (world report)',
    'tools/caregiver_exchange_preview.py': 'iterates a fresh 125 s run: 12,500 events, under the limit',
    'tools/caregiver_preview.py': 'iterates a fresh 50 s run: 5,000 events, under the limit',
    'tools/developmental_trial.py': 'dumps a list reset every 2,000 updates: 10,000 events, under the limit',
    'tools/caregiver_learning_trial.py': 'tape when responsive (record=True); replay lists are never trimmed',
    'tools/observer/baseline.py': 'len() only',
    # no line today; branch fix/ability_window_cost-2026-10-02 adds a counter of the same length
    'tools/gof_probe_harness.py': 'replaces tapes (sanitize_episode); len() only',
    'tools/prepare_decorated_shadow.py': 'copy.copy of a tape for its status path (shadow runs)',
    'tools/prepare_local_shadow.py': 'tape status path only',
    'tools/verify_continuous_world.py': "the world report's count",
}
PATTERN = re.compile(r"""(\[\s*['"]events['"]\s*\]|\.get\(\s*['"]events['"]).*""")


def test_no_unreviewed_reader_of_the_event_list():
    found = {}
    for path in ROOT.rglob('*.py'):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith('tests/'):
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for line in text.splitlines():
            if PATTERN.search(line) and re.search(r'caregiver|\bcg\b|self\.state|\bs\[', line):
                found.setdefault(rel, []).append(line.strip())
    unknown = {k: v for k, v in found.items() if k not in KNOWN_READERS}
    assert not unknown, unknown
