# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The model's gaze command beside the ledger, and the order centre's eye entries at an F1 switch
(the four F1 follow-ups of 2026-09-25, item 2; revised in the fourth review's fix round).

The ledger's eye row is written from the gaze the world SENT: the model's gaze command plus the saccade
generator's offset plus the gaze reflex's pull (0.7 of the way to a snap's target while it steers). The
four F1 follow-ups moved the efference order centre's eye entries at the switch by the change the replay
ring's recorded moments gave; that needed the ring refilled on merged code before the switch (training with
the eye bug live: the restart rule forbids it) and its row formula ignored the snap's blend (the fourth
review's findings 1 and 2). Since the fourth review's fix round (the coordinator's decision) the switch
leaves both running averages' eye entries as saved, and the ring's recorded commands mark the moments the
eye row's replay contract lets replay study (tests/test_eye_replay_contract.py). Pinned here:
  * the replay ring keeps the model's gaze command beside each outgoing order, back-filled with it,
    never across an episode reset, and a ring restored from a companion that did not hold it reads none;
  * the learner's step hands the command to the ring (a live world), and the trainer passes the command
    it sent, keeps it in the episode companion and clears it wherever it clears the ledger;
  * at the switch the order centre's eye entries are left as saved - even when the ring holds recorded
    commands, which the switch before the fix round moved them by - and the record says why;
  * end to end through the trainer (slow): a legacy micro-run's companion holds the commands, and a
    'physical' relaunch leaves the centre, writes the reason and the replay contract into its switch record,
    and stores its own moments under the eye row's replay contract, the first one after the switch without it.
"""
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.body_layout import names_for
from mimo_bridge.gaze_reflex import EL_RANGE
from mimo_bridge.lem_model import LemLite, command_for_encoder
from mimo_bridge.replay import ReplayMemory

NAMES = list(names_for("trunk_v1"))
H = len(NAMES)
GI = NAMES.index("gaze")
EYE_XY = [GI * 9 + 3, GI * 9 + 4]
PKG = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]


def _ledger(az, el, conv, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(B, H, 9, generator=g).numpy() * 10.0
    return LemLite.gaze_to_handle(o, np.stack([np.asarray(az, float), np.asarray(el, float)], -1), convention=conv)


def _ring(B=2, size=8):
    return ReplayMemory(["motor"], 4, n_handles=H, size_per_env=size, n_envs=B)


def _add(r, executed=None, ctrl=None, B=2):
    r.add({"motor": torch.zeros(B, 1, 4)}, torch.zeros(B, H, 10), {}, executed=executed, ctrl_gaze=ctrl)


# ------------------------------------------------------------------ the ring
def test_the_ring_keeps_the_controllers_eye_command_beside_each_outgoing_order():
    r = _ring()
    _add(r)
    e1 = torch.as_tensor(_ledger([0.1, -0.2], [0.3, -0.1], "legacy"), dtype=torch.float32)
    g1 = torch.tensor([[0.1, 0.25], [-0.2, -0.05]])
    _add(r, e1, g1)
    assert bool(r.has_executed[:, 0].all()) and bool(r.has_ctrl_gaze[:, 0].all())
    assert torch.equal(r.ctrl_gaze[:, 0], g1)
    assert not bool(r.has_ctrl_gaze[:, 1].any()), "the newest moment's outgoing command is not known yet"
    # an order without a command stores none
    _add(r, e1, None)
    assert bool(r.has_executed[:, 1].all()) and not bool(r.has_ctrl_gaze[:, 1].any())
    # never across an episode reset
    r.mark_reset()
    _add(r, e1, g1)
    assert not bool(r.has_executed[:, 2].any()) and not bool(r.has_ctrl_gaze[:, 2].any())
    with pytest.raises(ValueError, match="bodies"):
        _add(r, e1, torch.zeros(3, 2))


def test_a_ring_restored_without_the_record_reads_none_and_one_with_it_round_trips():
    from mimo_bridge import continuation_state as CS
    r = _ring()
    _add(r)
    _add(r, torch.as_tensor(_ledger([0.1, 0.1], [0.2, 0.2], "legacy"), dtype=torch.float32),
         torch.tensor([[0.1, 0.2], [0.1, 0.2]]))
    saved = CS.tree(vars(r))
    fresh = _ring()
    fresh.__dict__.update(saved)
    assert torch.equal(fresh.ctrl_gaze, r.ctrl_gaze) and torch.equal(fresh.has_ctrl_gaze, r.has_ctrl_gaze)
    old = {k: v for k, v in saved.items() if k not in ("ctrl_gaze", "has_ctrl_gaze")}
    fresh = _ring()
    fresh.__dict__.update(old)
    assert bool(fresh.has_executed[:, 0].all()) and not bool(fresh.has_ctrl_gaze.any())


def test_the_learner_hands_the_command_it_is_given_to_the_ring_in_a_live_world(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    torch.set_num_threads(1)
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True)
    m.enable_replay(1, size_per_env=32)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "w"), tag="_ecg", vision=False, seed=3)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        prev = prev_ctrl = None
        sent_ctrl = []
        for i in range(6):
            res = m.step(frames, dev, train=False, step_index=i, executed_action=prev,
                         executed_ctrl_gaze=prev_ctrl)
            gz = []
            for _ in range(2):
                frames = src.step(res.motor_for_action, res.gaze_out, None, render=False)
                gz.append(np.asarray(frames[0].meta.get("gaze_cmd"), dtype=np.float32))
            prev = LemLite.gaze_to_handle(res.motor_for_action, np.mean(gz, 0)[None])
            prev_ctrl = np.array(res.gaze_out, dtype=np.float32).reshape(1, 2)
            sent_ctrl.append(prev_ctrl.copy())
        r = m.replay
        assert bool(r.has_ctrl_gaze[0, :5].all()) and not bool(r.has_ctrl_gaze[0, 5])
        for t in range(5):
            assert torch.allclose(r.ctrl_gaze[0, t], torch.as_tensor(sent_ctrl[t][0]))
    finally:
        src.close()


def test_a_command_of_the_wrong_shape_is_left_out_and_said_once(caplog):
    """A recording must never stop training: a command that is not [B, 2] is not handed to the ring."""
    m = SimpleNamespace()
    exec_t = torch.zeros(2, H, 9)
    assert LemLite._ctrl_gaze_for_ring(m, None, exec_t, 2) == {}
    assert LemLite._ctrl_gaze_for_ring(m, np.zeros((2, 2)), None, 2) == {}
    got = LemLite._ctrl_gaze_for_ring(m, np.array([[0.1, 0.2], [0.3, 0.4]]), exec_t, 2)
    assert torch.equal(got["ctrl_gaze"], torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    with caplog.at_level("WARNING"):
        assert LemLite._ctrl_gaze_for_ring(m, np.zeros((3, 2)), exec_t, 2) == {}
        assert LemLite._ctrl_gaze_for_ring(m, np.zeros((3, 2)), exec_t, 2) == {}
    assert sum("not recorded beside the order" in r.getMessage() for r in caplog.records) == 1


def test_the_trainer_sends_keeps_and_clears_the_command_with_the_ledger():
    src = (PKG / "train_mimo.py").read_text(encoding="utf-8")
    # on its own line, so the memory repair's lines after the call merge beside it (the third review, F)
    assert ("executed_action=prev_executed,\n                             executed_ctrl_gaze=prev_ctrl_gaze,\n"
            "                             executed_order_id=prev_order_id)") in src.replace("\r\n", "\n")
    i_ledger = src.index("prev_executed = LemLite.gaze_to_handle(send_motor if limb_ledger is None else limb_ledger,")
    i_ctrl = src.index("prev_ctrl_gaze = (None if res.gaze_out is None")
    assert 0 < i_ctrl - i_ledger < 800, "the command is taken from the same burst as the ledger"
    assert "prev_ctrl_gaze=prev_ctrl_gaze," in src and "prev_ctrl_gaze=st.get('prev_ctrl_gaze')" in src
    assert "held_motor=prev_executed=prev_order_id=prev_ctrl_gaze=None" in src
    # every other place the ledger is set to None (the start, a phase rebuild, a scheduled reset) does it
    # to the command too
    lines = src.splitlines()
    cleared = [i for i, line in enumerate(lines) if line.strip() == "prev_executed = None"]
    assert len(cleared) == 3
    for i in cleared:
        assert any(x.strip() == "prev_ctrl_gaze = None" for x in lines[i:i + 7]), "line %d" % (i + 1)


# ------------------------------------------------------------------ the order centre at the switch
def _switching_learner(record=True, size=200, B=2, seed=4, world_el=0.12):
    """A 'physical' learner whose ring holds LEGACY ledger rows: each moment's controller command plus a
    world part of known mean elevation (world_el), with the command recorded beside it when `record`."""
    torch.set_num_threads(1)
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                body_layout="trunk_v1", efference_heads=True, eye_row_convention="physical")
    m.enable_replay(B, size_per_env=size)
    rng = np.random.default_rng(seed)
    az_c = rng.uniform(-0.4, 0.4, (B, size))
    el_c = rng.uniform(-0.5, 0.3, (B, size))
    w_el = world_el + rng.normal(0, 0.05, (B, size))
    w_az = rng.normal(0, 0.05, (B, size))
    for t in range(size):
        m.replay.executed[:, t] = torch.as_tensor(
            _ledger(az_c[:, t] + w_az[:, t], el_c[:, t] + w_el[:, t], "legacy", B=B, seed=t)).half()
        if record:
            m.replay.ctrl_gaze[:, t] = torch.as_tensor(np.stack([az_c[:, t], el_c[:, t]], -1), dtype=torch.float32)
    m.replay.has_executed.fill_(True)
    m.replay.has_ctrl_gaze.fill_(bool(record))
    m.replay.count = m.replay.ptr = size
    stored = command_for_encoder(m.replay.executed.float())[..., GI, 3:5].double()
    centre = torch.zeros(H * 9)
    centre[EYE_XY] = stored.reshape(-1, 2).mean(0).float()
    m.efference.order_centre.copy_(centre)
    m.efference.order_centre_weight = 1.0
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    return m, dict(az_c=az_c, el_c=el_c, w_el=w_el, w_az=w_az)


@pytest.mark.parametrize("record", [True, False])
def test_at_the_switch_the_order_centre_is_left_as_saved_with_the_reason(record):
    """The fourth review's fix round (the coordinator's decision on its blocker 1; its finding 2): the switch no
    longer reads the ring's recorded commands to move the centre - with them (record=True, which the switch before
    the fix round moved the centre by, off by the reflex's blend during snaps) or without them (the deploy's ring,
    saved by production's code) the eye entries stay as saved and the record says why. The stored rows are still
    rewritten, exactly."""
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, _ = _switching_learner(record=record)
    before = m.efference.order_centre.detach().clone()
    rows = m.replay.executed.clone()
    _, done = switch_stored_eye_rows(m, None)
    assert torch.equal(m.efference.order_centre, before), "nothing in the centre moves"
    rec = m.motor_interface_conversion["order_centre_eye_shift"]
    assert rec["shift"] is None and rec["entries"] == EYE_XY
    assert np.allclose(rec["centre"], before[EYE_XY].numpy(), atol=0)
    assert rec["left_as_saved"] == LemLite.CENTRE_LEFT_AS_SAVED
    assert "training with the eye bug live" in rec["left_as_saved"] and "1/2,000 a step" in rec["left_as_saved"]
    assert done["replay.executed"] == 400
    assert torch.equal(m.replay.executed[..., GI, 3:5], -rows[..., GI, 3:5])
    other = torch.ones(9, dtype=torch.bool)
    other[3:5] = False
    assert torch.equal(m.replay.executed[..., GI, :][..., other], rows[..., GI, :][..., other])


def test_a_centre_never_set_is_not_moved():
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, _ = _switching_learner()
    m.efference.order_centre_weight = 0.0
    before = m.efference.order_centre.detach().clone()
    switch_stored_eye_rows(m, None)
    assert torch.equal(m.efference.order_centre, before)
    assert "order_centre_eye_shift" not in m.motor_interface_conversion


# ------------------------------------------------------------------ end to end through the trainer
def _micro_run(out, extra, steps):
    from mimo_bridge import train_mimo as TM
    os.makedirs(out, exist_ok=True)
    argv = ["--n-envs", "2", "--no-vision", "--minutes", "5", "--scene", "reach", "--out", out,
            "--video-every-min", "0", "--replay", "on", "--replay-size", "64", "--curriculum", "on",
            "--next-frame-loss", "on", "--intention", "off", "--efference-heads", "on",
            "--report-every", "10", "--max-steps", str(steps)] + extra
    assert TM.main(argv) == 0, "trainer micro-run failed"
    return [json.loads(x) for x in open(os.path.join(out, "replay_training_stats.jsonl"), encoding="utf-8")
            if x.strip()]


@pytest.mark.slow
def test_through_the_trainer_a_switch_leaves_the_centre_and_stores_its_moments_under_the_eye_contract(tmp_path):
    from mimo_bridge import continuation_state as CS
    import mimo_bridge.limb_target_control as LTC
    EYE_ROW_CONTRACT_BIT = getattr(LTC, "EYE_ROW_CONTRACT_BIT", 16)     # by name: the code before the fix fails below
    # the micro-runs' checkpoints under pytest's own folder (PYTEST_DEBUG_TEMPROOT or --basetemp puts it on
    # D:, never C:), removed at the end
    tmp = tmp_path / "ecg_run"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        _micro_run(str(tmp / "legacy"), [], 40)
        ck = sorted((tmp / "legacy" / "checkpoints").glob("ckpt_step_*"))[-1]
        st = CS.load(str(ck))
        rp = st["learner"]["objects"]["replay"]
        n = int(rp["count"])
        lab = rp["has_executed"][:, :n]
        assert int(lab.sum()) >= 60 and torch.equal(rp["has_ctrl_gaze"][:, :n], lab)
        assert not bool((rp["action_contract"][:, :n] & EYE_ROW_CONTRACT_BIT).any()), "legacy: no eye bit"
        assert st["trainer"]["prev_ctrl_gaze"] is not None
        recs = _micro_run(str(tmp / "physical"), ["--resume", str(ck), "--eye-row-convention", "physical"], 43)
        sw = [r for r in recs if r.get("type") == "motor_interface_stored_rows"]
        assert len(sw) == 1
        shift = sw[0]["order_centre_eye_shift"]
        assert shift["shift"] is None and "left as saved at the switch" in shift["left_as_saved"]
        rc = sw[0]["replay_contract"]
        assert rc["required"] & EYE_ROW_CONTRACT_BIT and rc["stored_moments_under_it"] == 0
        ck2 = sorted((tmp / "physical" / "checkpoints").glob("ckpt_step_*"))[-1]
        rp2 = CS.load(str(ck2))["learner"]["objects"]["replay"]
        size, ptr, count = int(rp2["size"]), int(rp2["ptr"]), int(rp2["count"])
        ac, hx = rp2["action_contract"], rp2["has_executed"]
        # the slots written after the switch: the last 3 adds (steps 40, 41, 42) back-filled slots ptr-4..ptr-2
        # (the newest, ptr-1, has no outgoing order yet); the first of them is the last moment before the switch,
        # back-filled without the command (the trainer drops it at the switch), the others with it
        slots = [(ptr - k) % size for k in (4, 3, 2)]
        assert bool(hx[:, slots].all())
        assert not bool((ac[:, slots[0]] & EYE_ROW_CONTRACT_BIT).any())
        assert bool((ac[:, slots[1:]] & EYE_ROW_CONTRACT_BIT).all())
        assert count >= 43
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
