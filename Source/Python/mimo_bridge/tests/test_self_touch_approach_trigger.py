# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The intervention must not trigger on receding or already-touched targets."""
import importlib.util
from pathlib import Path


def load(name):
    path=Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def test_approach_trigger_requires_prior_closing_motion_and_no_contact():
    trigger=load('probe_self_touch_approach_choice').trigger
    assert trigger(.01,.02,.015,True)
    assert not trigger(.02,.03,.015,True)
    assert not trigger(.01,.009,.015,True)
    assert not trigger(.01,.02,.015,False)
    assert not trigger(-.001,.02,.015,True)
    assert not trigger(float('nan'),.02,.015,True)


def test_physics_deadline_is_checked_inside_expensive_work(monkeypatch):
    import pytest
    audit=load('audit_self_touch_approach')
    monkeypatch.setattr(audit,'DEADLINE',100.)
    monkeypatch.setattr(audit.time,'time',lambda:100.01)
    with pytest.raises(audit.DeadlineReached):audit.checkpoint_progress('kinematic search')


def test_first_hand_endpoint_excludes_other_sources_and_structural_contacts():
    group_onset=load('calibrate_first_hand_touch').group_onset
    group=dict(hand_bodies=['right_hand','right_thbase'],target_body='upper_body')
    event=dict(pair=['right_thbase','upper_body'],structural=False)
    assert group_onset({'self_contact_onsets':[event]},group)
    assert not group_onset({'self_contact_onsets':[dict(event,structural=True)]},group)
    assert not group_onset({'self_contact_onsets':[dict(event,pair=['left_hand','upper_body'])]},group)
    assert not group_onset({'self_contact_onsets':[dict(event,pair=['right_hand','lower_body'])]},group)
