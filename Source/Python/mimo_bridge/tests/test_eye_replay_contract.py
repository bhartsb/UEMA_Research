# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The eye row's replay contract (the fourth review of the eye fixes, fix round; the coordinator's decision on its
blocker 1: the deploy must not need the replay ring refilled on merged code, which would be training with the eye bug
live).

Under --eye-row-convention physical replay studies only windows whose every step-from moment was recorded under that
convention with the model's gaze command beside its order (limb_target_control.EYE_ROW_CONTRACT_BIT, set by
LemLite._eye_contract_bit); the moments stored before the switch are skipped, since their stored memory and self
states were encoded from old-convention rows, which no rewrite reaches. Pinned here:
  * the contract: the eye bit under 'physical' only; under 'legacy' every contract is the one of before;
  * the learner's step sets the bit on a back-filled moment only under 'physical' and only when the command
    arrives with the order (a live world); under 'legacy' the stored contract is the one of before;
  * after a switch replay studies nothing until a whole window of post-switch moments exists - the first one
    after ctx_len + 2 post-switch updates - and then only post-switch windows; a 'legacy' learner on the same
    ring studies the old moments as before;
  * the trainer drops the last pre-switch transition's command at the switch, so that order carries no bit.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

import mimo_bridge.limb_target_control as LTC
from mimo_bridge.limb_target_control import (CALF_CONTRACT_BIT, CHEST_CONTRACT_BIT, POSITION_ROWS_CONTRACT_BIT,
                                             replay_contract)
from mimo_bridge.lem_model import LemLite

# 16 (the fix's value); read by name so the file also runs on the code before the fix, where the tests below fail
# on what replay studies, not on a missing name
EYE_ROW_CONTRACT_BIT = getattr(LTC, "EYE_ROW_CONTRACT_BIT", 16)

PKG = Path(__file__).resolve().parents[1]


def _old_formula(hand, parent, rows):
    """replay_contract as it was before the eye row's bit (copied from the limb fixes' commit)."""
    bits = 0 if hand == "world" else 1
    if parent in ("calf", "calf_chest"):
        bits |= CALF_CONTRACT_BIT
    if parent == "calf_chest":
        bits |= CHEST_CONTRACT_BIT
    if rows == "no_head_pelvis":
        bits |= POSITION_ROWS_CONTRACT_BIT
    return bits or None


@pytest.mark.parametrize("hand", ["world", "wrist_relative_v1"])
@pytest.mark.parametrize("parent", ["off", "calf", "calf_chest"])
@pytest.mark.parametrize("rows", ["all", "no_head_pelvis"])
def test_the_eye_bit_is_in_the_contract_under_physical_only(hand, parent, rows):
    assert replay_contract(hand, parent, rows) == _old_formula(hand, parent, rows)
    assert replay_contract(hand, parent, rows, "legacy") == _old_formula(hand, parent, rows)
    assert replay_contract(hand, parent, rows, "physical") == (_old_formula(hand, parent, rows) or 0) | EYE_ROW_CONTRACT_BIT
    assert EYE_ROW_CONTRACT_BIT not in (1, CALF_CONTRACT_BIT, CHEST_CONTRACT_BIT, POSITION_ROWS_CONTRACT_BIT)


def test_the_learner_sets_the_bit_under_physical_when_the_command_arrives_in_a_live_world(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    torch.set_num_threads(1)
    got = {}
    for conv in ("legacy", "physical"):
        m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                    eye_row_convention=conv)
        m.enable_replay(1, size_per_env=32)
        src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / conv), tag="_erc" + conv, vision=False, seed=3)
        try:
            frames = src.reset()
            dev = dict(m.dev_limits(10 ** 9))
            prev = prev_ctrl = None
            for i in range(7):
                res = m.step(frames, dev, train=False, step_index=i, executed_action=prev,
                             executed_ctrl_gaze=None if i == 3 else prev_ctrl)   # step 3's order arrives alone
                gz = []
                for _ in range(2):
                    frames = src.step(res.motor_for_action, res.gaze_out, None, render=False)
                    gz.append(np.asarray(frames[0].meta.get("gaze_cmd"), dtype=np.float32))
                prev = LemLite.gaze_to_handle(res.motor_for_action, np.mean(gz, 0)[None], convention=conv)
                prev_ctrl = np.array(res.gaze_out, dtype=np.float32).reshape(1, 2)
        finally:
            src.close()
        r = m.replay
        got[conv] = [int(r.action_contract[0, t]) for t in range(6)], [bool(r.has_executed[0, t]) for t in range(6)]
    bits_legacy, lab = got["legacy"]
    bits_physical, lab_p = got["physical"]
    assert lab == lab_p == [True] * 6
    assert all(b & EYE_ROW_CONTRACT_BIT == 0 for b in bits_legacy), "under 'legacy' no moment carries the eye bit"
    # moment 2 was back-filled at step 3, whose order arrived without the command
    assert [bool(b & EYE_ROW_CONTRACT_BIT) for b in bits_physical] == [True, True, False, True, True, True]
    # every other bit is the one the world's reports give, the same under both conventions
    assert [b & ~EYE_ROW_CONTRACT_BIT for b in bits_physical] == bits_legacy


# ------------------------------------------------------------------ replay after a switch
_MODELS = {}


def _learner(conv):
    if conv not in _MODELS:
        torch.set_num_threads(1)
        _MODELS[conv] = LemLite(device="cpu", seed=0, imagination=False, eye_row_convention=conv)
    return _MODELS[conv]


def _ring_learner(conv, size=40, pre=20, post=0, B=2):
    """A learner whose (fresh) ring holds `pre` moments recorded before a switch (no eye bit) and `post` after it,
    as the trainer stores them: the first post-switch add back-fills the last pre-switch moment WITHOUT the
    command (the trainer drops it at the switch), every later add back-fills with it, the bit set by the
    learner's own _eye_contract_bit. Each moment's tokens carry its index, so a sampled window says which
    moments it read."""
    m = _learner(conv)
    m.enable_replay(B, size_per_env=size)
    nh = len(m.handle_names)
    order = torch.zeros(B, nh, 9)
    order[..., 6] = 1.0
    ctrl = torch.zeros(B, 2)
    for t in range(pre + post):
        toks = {mm: torch.full((B, 1, m.d_model), float(t)) for mm in m.active}
        cg = {} if t <= pre else {"ctrl_gaze": ctrl}
        m.replay.add(toks, torch.zeros(B, nh, 10), {"rom_limit": 1.0, "strength_limit": 1.0},
                     executed=order if t > 0 else None, action_contract=_bit(m, cg), **cg)
    return m


def _bit(m, cg):
    """The learner's own _eye_contract_bit; on the code before the fix, which has none, the contract the fixed
    learner stores (the live-world test above pins that), so the ring is the same and only replay differs."""
    own = getattr(m, "_eye_contract_bit", None)
    if own is not None:
        return own(cg)
    return EYE_ROW_CONTRACT_BIT if cg and m.eye_row_convention == "physical" else 0


def _windows_read(m, batch=16):
    """Run replay_update and return the moment indices (token stamps) of every window it sampled, or None when
    it studied nothing."""
    seen = []
    real = m.replay.sample_windows

    def spy(*a, **k):
        w = real(*a, **k)
        if w is not None:
            seen.append(w["tokens"][:, :, 0, 0].clone())
        return w
    m.replay.sample_windows = spy
    try:
        out = m.replay_update(batch=batch)
    finally:
        m.replay.sample_windows = real
    return None if out is None else torch.cat(seen)


def test_after_a_switch_replay_studies_no_moment_stored_before_it():
    m = _ring_learner("physical", pre=20, post=0)
    assert _windows_read(m) is None, "only pre-switch moments: nothing is studied"
    # a 'legacy' learner on the same ring studies the old moments as before
    legacy = _ring_learner("legacy", pre=20, post=0)
    assert _windows_read(legacy) is not None


def test_replay_waits_for_a_whole_window_of_post_switch_moments_and_then_reads_only_them():
    L = _learner("physical").ctx_len
    first = None
    for post in range(1, L + 6):
        m = _ring_learner("physical", size=60, pre=20, post=post)
        w = _windows_read(m)
        if w is not None:
            first = post
            # moments 20.. are the post-switch ones; moment 19, the last before the switch, was back-filled
            # without the command at the first post-switch update and carries no bit
            assert bool((w >= 20).all()), "a window read a moment stored before the switch"
            break
    # a window's L step-from moments (post-switch moments 0..L-1, each back-filled with the command at the next
    # update) and its target, which is never the newest moment: L + 2 post-switch updates
    assert first == L + 2
    m = _ring_learner("physical", size=60, pre=20, post=L + 20)
    w = _windows_read(m, batch=64)
    assert bool((w >= 20).all()) and int(w.max()) <= 20 + L + 20 - 2
    # every window is L + 1 consecutive moments
    assert bool(((w[:, 1:] - w[:, :-1]) == 1).all())


def test_the_trainer_drops_the_pre_switch_command_at_the_switch():
    src = (PKG / "train_mimo.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    i = src.index("prev_executed, _eye_rows = switch_stored_eye_rows(model, prev_executed)")
    block = src[i:i + 600]
    assert "if _eye_rows is not None:" in block and "prev_ctrl_gaze = None" in block
    assert block.index("prev_ctrl_gaze = None") < block.index("sf.write(")
