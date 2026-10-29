# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Selecting a probe body must keep its saved history, including channel-first data."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mimo_bridge.tools import gof_probe_harness as H


class _ReloadModel:
    """LemLite's per-body reload, reduced to what restore_runtime_body reads. The checkpoint holds
    no contact head, so the loader reports it reinitialised and leaves it as it is
    (lem/training/checkpoint.py: a side module absent from the checkpoint is only listed).
    `redraw` stands for a loader that re-creates the part at every reload instead."""

    def __init__(self, reports, redraw=False):
        gen = torch.Generator().manual_seed(0)
        self.contact_head = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.contact_head.weight.copy_(torch.randn(2, 3, generator=gen))
        # a running statistic not yet measured: NaN is unchanged when it stays NaN
        self.contact_head.register_buffer("centre", torch.tensor([float("nan"), 0.5]), persistent=False)
        self.reports, self.redraw, self.loads = list(reports), redraw, 0

    def _side_modules(self):
        return {"contact_head": self.contact_head}

    def load_checkpoint(self, path, load_optimizer=True):
        self.loads += 1
        if self.redraw:
            with torch.no_grad():
                self.contact_head.weight.add_(1.0)
        return {"reinitialized_components": list(self.reports)}


def _frozen(model, first_load):
    return H.Frozen(model=model, args=SimpleNamespace(), info={"checkpoint": "ckpt"}, limits={},
                    limits_source="test", load=first_load, build_s=0.0, load_s=0.0)


def test_a_body_reload_keeps_the_part_the_first_load_started_fresh():
    """test_gof_b4_checkpoint builds a contact head that checkpoint 344,400 lacks: its first load
    reports the head reinitialised, and so does every per-body reload. The reload is accepted for
    exactly that declared part, and the part leaves it unchanged, so every body is read by one
    head."""
    m = _ReloadModel(["contact_head"])
    head, weight = m.contact_head, m.contact_head.weight.detach().clone()
    frozen = _frozen(m, {"reinitialized": ["contact_head"]})
    for body in range(3):
        H.restore_runtime_body(frozen, 3, body)
    assert m.loads == 3 and m.contact_head is head
    assert torch.equal(m.contact_head.weight, weight)


def test_a_body_reload_refuses_every_part_the_first_load_did_not_start_fresh():
    # load_frozen refuses any reinitialisation, so its Frozen declares none: every probe keeps the
    # rule that a reload which reinitialises anything is refused
    for first_load in ({"reinitialized": []}, {}):
        with pytest.raises(H.RefusedCheckpoint, match="contact_head"):
            H.restore_runtime_body(_frozen(_ReloadModel(["contact_head"]), first_load), 2, 0)
    # a trained part coming back fresh is refused by name, beside a declared fresh one
    with pytest.raises(H.RefusedCheckpoint, match="encoder:vision"):
        H.restore_runtime_body(_frozen(_ReloadModel(["contact_head", "encoder:vision"]),
                                       {"reinitialized": ["contact_head"]}), 2, 0)


def test_a_body_reload_that_re_creates_a_declared_fresh_part_is_refused():
    """A declared part is accepted only while the reload leaves every one of its tensors as it
    was: re-drawn, each body would be read by another head."""
    m = _ReloadModel(["contact_head"], redraw=True)
    with pytest.raises(H.RefusedCheckpoint, match="re-created"):
        H.restore_runtime_body(_frozen(m, {"reinitialized": ["contact_head"]}), 2, 0)
    replaced = _ReloadModel(["contact_head"])
    real_load = replaced.load_checkpoint

    def replacing_load(path, load_optimizer=True):
        replaced.contact_head = torch.nn.Linear(3, 2)
        return real_load(path, load_optimizer)
    replaced.load_checkpoint = replacing_load
    with pytest.raises(H.RefusedCheckpoint, match="re-created"):
        H.restore_runtime_body(_frozen(replaced, {"reinitialized": ["contact_head"]}), 2, 0)
    unknown = _ReloadModel(["gaze_head"])
    with pytest.raises(H.RefusedCheckpoint, match="gaze_head"):
        H.restore_runtime_body(_frozen(unknown, {"reinitialized": ["gaze_head"]}), 2, 0)


def test_named_body_axes_win_when_the_channel_or_control_count_matches_the_batch():
    channels = torch.arange(64).reshape(8, 8)
    state = dict(_goal_ring_ch=deque([channels], maxlen=200),
                 _contact_habit=[{'body': i} for i in range(8)], _gl_goal_age=list(range(8)))
    selected = H.slice_batch(state, 8, 3)
    assert torch.equal(selected['_goal_ring_ch'][0], channels[:, 3:4])
    assert selected['_contact_habit'] == [{'body': 3}]
    assert selected['_gl_goal_age'] == [3]
    assert state['_goal_ring_ch'][0].shape == (8, 8)
    entity = dict(l2_p_fast=channels, l2_ncov=channels, l2_nvvar=torch.arange(8))
    chosen = H.slice_entity_body(entity, 8, 3)
    assert torch.equal(chosen['l2_p_fast'], channels[3:4])
    assert torch.equal(chosen['l2_ncov'], channels)
    assert torch.equal(chosen['l2_nvvar'], torch.arange(8))


CKPT = Path(__file__).resolve().parents[4] / 'Saved/gof_takeover_20260923/proof_400/experiments/candidate/checkpoints/ckpt_step_344800'


@pytest.mark.slow
@pytest.mark.skipif(not CKPT.is_dir(), reason='local takeover proof checkpoint unavailable')
def test_saved_upgrade_keeps_each_bodys_goal_and_entity_history_and_replays():
    from mimo_bridge import copied_execution as CE
    f = H.load_frozen(CKPT)
    ep, _ = H.load_episode(f)
    assert len(ep['worlds']) == 8 and not f.load['reinitialized']
    for b in (0, 1):
        s = H.Session(f, episode=ep, body=b, gaze_reflex=False, saccades=False)
        try:
            m = f.model
            expected = ep['learner']['data']['_goal_ring_ch'][-1][:, b:b + 1]
            torch.testing.assert_close(m._goal_ring_ch[-1], expected, rtol=0, atol=0, equal_nan=True)
            assert m._goal_ring_ch[-1].shape[1] == 1
            count, cleared = len(m.goal_ring), m._rank_ring_cleared
            frames = int(m._entity_self_saved['frames'].item())
            nulls = m._entity_self_saved['l2_ncov'].clone()
            s.observe()
            assert len(m.goal_ring) == count and m._rank_ring_cleared == cleared
            assert m._entity_self.n_bodies == 1 and int(m._entity_self.frames.item()) == frames + 1
            assert m._entity_self.l2_ncov.shape == nulls.shape
            snap, order = s.snapshot(), s.hold_order()
            outputs = []
            for _ in range(2):
                s.restore(snap)
                s.act(order)
                outputs.append(CE.output_record(m, s.observe()))
            assert CE.equal_tree(outputs[0], outputs[1]), CE.first_difference(outputs[0], outputs[1])
        finally:
            s.close()
