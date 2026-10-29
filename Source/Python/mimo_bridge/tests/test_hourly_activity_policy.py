# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exercise the actual watcher without waiting an hour or starting a trainer."""
import collections
import json
from pathlib import Path
import runpy
import shutil
import sys
import time

import pytest


WATCHER=Path(__file__).resolve().parents[4]/'docs/consequence_repair_2026-09-09/hourly_watch.py'


# The watcher stops a run for a LOW RESOURCE before it looks at any training
# number, which is correct behaviour and wrong for these tests: on a machine with
# under 8 GB free on C: every test below stopped for the disk instead of the rule
# it names, so the file passed or failed on the developer's disk space rather
# than on the watcher's logic. Measured 2026-09-16 on a box with 7.3 GB free:
# 5 of 6 parameters failed, and the same 5 failed with every source change
# reverted. Hold the resources healthy here; test_a_low_disk_stops_the_run below
# covers the rule these stubs switch off.
HEALTHY_DISK = collections.namedtuple('usage','total used free')(1e12, 0.0, 500e9)


def healthy_resources(monkeypatch):
    monkeypatch.setattr(shutil,'disk_usage',lambda _p: HEALTHY_DISK)
    try:
        import psutil
    except ImportError:
        return
    monkeypatch.setattr(psutil,'virtual_memory',
                        lambda: collections.namedtuple('vm','available')(64e9))


def watch(tmp_path,monkeypatch,resources='healthy',**changes):
    if resources=='healthy':
        healthy_resources(monkeypatch)
    row=dict(type='mimo',step=500,timestamp=10000,elapsed_s=7200,nan_steps=0,
             token_temporal_std=dict(vision=.6,tactile=.6),
             causal_pe_per_modality=dict(vision=.01,tactile=.04,proprioception=.005),
             bout_body_speed=.00154,follow_bout=.1,task_hand_fov=.001)
    row.update(changes)
    (tmp_path/'args.json').write_text('{}')
    (tmp_path/'replay_training_stats.jsonl').write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(sys,'argv',[str(WATCHER),str(tmp_path),'3'])
    monkeypatch.setattr(time,'sleep',lambda _:None)
    monkeypatch.setattr(time,'time',lambda:10000)
    runpy.run_path(str(WATCHER),run_name='__main__')
    control=tmp_path/'control.json'
    return json.loads(control.read_text()) if control.exists() else None


@pytest.mark.parametrize('speed',[.00154,0.0])
def test_repeated_low_activity_reports_but_does_not_stop(tmp_path,monkeypatch,capsys,speed):
    assert watch(tmp_path,monkeypatch,bout_body_speed=speed) is None
    output=capsys.readouterr().out
    assert output.count('ACTIVITY ALERT:')==3
    assert 'cause untested' in output and 'Training continues' in output
    assert not (tmp_path/'stop_reason.json').exists()


@pytest.mark.parametrize('changes,reason',[
    (dict(nan_steps=1),'non-finite'),
    (dict(token_temporal_std=dict(vision=.01,tactile=.6)),'vision variance'),
    (dict(token_temporal_std=dict(vision=.6,tactile=.01)),'tactile variance'),
    (dict(follow_bout=-.1),'bout obedience'),
    (dict(causal_pe_per_modality=dict(vision=.1,tactile=.04)),'vision causal'),
    (dict(causal_pe_per_modality=dict(vision=.01,tactile=.2)),'tactile causal'),
])
def test_low_activity_does_not_disable_other_repair_stops(tmp_path,monkeypatch,changes,reason):
    assert watch(tmp_path,monkeypatch,**changes)['cmd']=='snapshot_and_stop'
    receipt=json.loads((tmp_path/'stop_reason.json').read_text())
    assert receipt['resume_policy']=='repair_required' and reason in receipt['reason']


def test_a_low_disk_stops_the_run_before_any_training_number_is_read(tmp_path,monkeypatch):
    """The rule the stubs above switch off, exercised on its own.

    Free space is forced to 1 GB against the watcher's 8 GB line, and the
    training row is left completely healthy, so the only thing that can stop the
    run is the disk.
    """
    monkeypatch.setattr(shutil,'disk_usage',
                        lambda _p: HEALTHY_DISK._replace(free=1e9))
    control=watch(tmp_path,monkeypatch,resources='real')
    assert control['cmd']=='snapshot_and_stop'
    receipt=json.loads((tmp_path/'stop_reason.json').read_text())
    assert 'free disk' in receipt['reason'] and '8 GB line' in receipt['reason']
    assert receipt['resume_policy']=='repair_required'


def test_a_healthy_disk_does_not_stop_a_healthy_run(tmp_path,monkeypatch):
    """The other side of the same rule, so the stub cannot hide a false stop."""
    assert watch(tmp_path,monkeypatch) is None
    assert not (tmp_path/'stop_reason.json').exists()
