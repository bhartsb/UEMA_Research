# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What a motor interface switch converts, and what it leaves (LemLite._convert_order_averages,
LemLite.convert_stored_eye_rows, train_mimo.switch_stored_eye_rows).

The efference body head reads every order field minus a running mean over 2,000 lived moments
(lem/model/efference.py, order_centre, saved in runtime_state.pt). At the deploy of
--executed-fields-only on every radius becomes 0 while the saved mean holds the old radii (0.50 on
checkpoint 401,750's ring, 1.7 standard deviations): the radius entries are set to 0.

At the deploy of --eye-row-convention physical (the deploy branch's review, finding 3) the controller's
part of the LEDGER'S eye rows keeps its numbers: the controller writes the same row, only the command
sent to the eye servo flips, and gaze_to_handle writes the row of that command, which is the
controller's row again. In the world the ledger is written from the gaze sent, which keeps only the
command's share (all of it with no reflex snap, 0.3 of it while a snap steers) and adds the saccade
generator's offset, the snap's 0.7 of its target and the clip at the eye's range; the switch writes those
mirrored (not in these tests: they have no world; tests/test_motor_fixes_eye_row_split.py has the world's
arithmetic). So the LOAD leaves the running
mean of the orders' eye entries as saved, and since the fourth review's fix round so does the switch after it
(tests/test_order_centre_eye_shift.py; LemLite.CENTRE_LEFT_AS_SAVED). What changes is the meaning of every row
STORED before the switch: it
is the row of the command it sent in the OLD convention, and read in the new one it names the mirrored
elevation. Those rows are rewritten as the new convention's row for the same command (quaternion x and
y negated), which is exact.

Pinned here:
  * the ledger row equals the controller's row under both conventions (the review's probe), so the
    centre's eye entries stay;
  * the radius conversion is exact, a checkpoint round trip converts exactly those entries and
    nothing else, and with no switch the centre is bit-identical and no record is written;
  * the stored rows (replay ring, the heads' gaze history and tenure order, the imagination chain's
    recent orders, the grounded self's actions and rehearsal windows, the trainer's last ledger) are
    rewritten exactly at a convention switch, the gaze copy then reads the elevation each stored
    moment sent, and nothing is touched without the switch;
  * only an average THIS load restored is converted, and the grounded self's command calibration gets
    the radius conversion when the load read it.
"""
import math
import shutil
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.model.efference import CENTRE_HORIZON_STEPS, gaze_order_features
from mimo_bridge.body_layout import names_for
from mimo_bridge.lem_model import LemLite, command_for_encoder

NAMES = list(names_for("trunk_v1"))
H = len(NAMES)
GI = NAMES.index("gaze")
RADIUS = [h * 9 + 7 for h in range(H)]
EYE_XY = [GI * 9 + 3, GI * 9 + 4]
PKG = Path(__file__).resolve().parents[1]


def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1", efference_heads=True, **kw)


def _orders(n, B=2, seed=0):
    """n moments of B bodies' orders in encoder units: positions / 200, a unit quaternion, a radius
    in 0..1 (mean 0.5, the ring's measured mean) and a strength."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(n, B, H, 9, generator=g) * 0.1
    q = torch.randn(n, B, H, 4, generator=g)
    o[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    o[..., 7] = torch.rand(n, B, H, generator=g)
    o[..., 8] = torch.rand(n, B, H, generator=g)
    return o


def _centre_over(orders):
    m = _lite()
    lat = torch.zeros(orders.shape[1], m.n_mod, m.d_model)
    m.efference.ensure_centres(lat, orders[0])
    for o in orders:
        m.efference.update_centres(lat, o)
    return m.efference.order_centre.detach().clone(), float(m.efference.order_centre_weight)


def test_the_ledgers_eye_row_is_the_controllers_row_under_both_conventions():
    """The deploy review's probe: the weights are the same at the switch, so the controller writes the
    same eye row R; the servo is sent +pitch(R) under 'legacy' and -pitch(R) under 'physical', and the
    ledger writes the row of what was sent in its convention - R again, under both."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        az, pitch = rng.uniform(-0.7, 0.7), rng.uniform(-0.5, 0.5)
        ctrl = LemLite.gaze_to_handle(np.zeros((1, H, 9)), np.array([[az, pitch]]), convention="legacy")
        rows = []
        for conv in ("legacy", "physical"):
            m = SimpleNamespace(eye_row_convention=conv, GAZE_IDX=GI)
            sent = LemLite._gaze_from_handle(m, torch.as_tensor(ctrl, dtype=torch.float32))[0]
            rows.append(LemLite.gaze_to_handle(ctrl, sent[None], convention=conv)[0, GI, 3:7])
            assert abs(float(sent[1]) - (pitch if conv == "legacy" else -pitch)) < 1e-5
        assert np.allclose(rows[0], rows[1], atol=1e-6) and np.allclose(rows[0], ctrl[0, GI, 3:7], atol=1e-6)


@pytest.mark.parametrize("physical,fields", [(True, False), (False, True), (True, True)])
def test_the_radius_entries_go_to_the_mean_of_zeros_and_the_eye_entries_stay(physical, fields):
    orders = _orders(40)
    old, w_old = _centre_over(orders)
    zeroed = orders.clone()
    zeroed[..., 7] = 0.0
    new, w_new = _centre_over(zeroed)
    assert w_old == w_new
    m = _lite(eye_row_convention="physical" if physical else "legacy",
              executed_fields_only="on" if fields else "off")
    m.efference.order_centre.copy_(old)
    m.efference.order_centre_weight = w_old
    m.motor_interface_change = m._motor_interface_changes({})
    rec = m._convert_order_averages(efference_restored=True, self_state_loaded=False)
    got = m.efference.order_centre
    changed = RADIUS if fields else []
    assert torch.allclose(got[changed], new[changed], atol=1e-7, rtol=0)
    keep = torch.ones(H * 9, dtype=torch.bool)
    keep[changed] = False
    assert torch.equal(got[keep], old[keep]), "the eye entries (and everything else) stay as saved"
    a = rec["averages"]["efference.order_centre"]
    assert a["radius_set_to_0"] == fields and "eye_row_x_y_negated" not in a
    assert abs(a["radius_mean_before"] - float(old[RADIUS].mean())) < 1e-6
    assert any("eye row's entries" in s for s in rec["left_as_saved"]) == physical


def _save(m, tmp):
    from lem.training.checkpoint import CheckpointManager
    m.checkpoints = CheckpointManager(m.config, str(tmp))
    return m.save_checkpoint(1000)


@pytest.fixture
def saved_legacy_checkpoint(tmp_path):
    # under pytest's own folder (PYTEST_DEBUG_TEMPROOT or --basetemp puts it on D:), removed at the end
    tmp = tmp_path / "order_avg"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        m = _lite()
        orders = _orders(30, seed=3)
        lat = torch.randn(2, m.n_mod, m.d_model)
        m.efference.ensure_centres(lat, orders[0])
        for o in orders:
            m.efference.update_centres(lat, o)
        centre = m.efference.order_centre.detach().clone()
        latent_centre = m.efference.centre_mean.detach().clone()
        weight = float(m.efference.order_centre_weight)
        path = _save(m, tmp)
        yield path, centre, latent_centre, weight
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("physical,fields", [(False, False), (True, False), (False, True), (True, True)])
def test_a_resume_converts_exactly_the_switched_entries(saved_legacy_checkpoint, physical, fields):
    path, centre, latent_centre, weight = saved_legacy_checkpoint
    m = _lite(eye_row_convention="physical" if physical else "legacy",
              executed_fields_only="on" if fields else "off")
    m.load_checkpoint(path, load_optimizer=False)
    got = m.efference.order_centre
    assert torch.equal(m.efference.centre_mean, latent_centre), "the latent's centre is not an order average"
    assert float(m.efference.order_centre_weight) == weight
    want = centre.clone()
    if fields:
        want[RADIUS] = 0.0
    assert torch.equal(got, want)
    if not (physical or fields):
        assert m.motor_interface_change is None and m.motor_interface_conversion is None
    else:
        rec = m.motor_interface_conversion
        assert set(rec["changes"]) == ({"eye_row_convention"} if physical else set()) | (
            {"executed_fields_only"} if fields else set())
        assert set(rec["averages"]) == {"efference.order_centre"}
        assert any("decoder ignores" in s for s in rec["left_as_saved"])


def test_a_resume_without_the_receipt_refuses_a_switch_rather_than_mirror_rows_back(saved_legacy_checkpoint):
    """The deploy branch's second review, point 7: the switch's trigger is the receipt's stamp. Without the
    receipt, a checkpoint trained under 'physical' would read as 'legacy' and its stored rows would be
    mirrored back; the load refuses instead, and a launch with the defaults still loads it."""
    path, centre, _, _ = saved_legacy_checkpoint
    Path(path, "model_receipt.json").unlink()
    for kw in ({"eye_row_convention": "physical"}, {"executed_fields_only": "on"}):
        m = _lite(**kw)
        with pytest.raises(ValueError, match="no model_receipt.json"):
            m.load_checkpoint(path, load_optimizer=False)
    m = _lite()
    m.load_checkpoint(path, load_optimizer=False)
    assert m.motor_interface_change is None and torch.equal(m.efference.order_centre, centre)


def test_the_load_says_the_stored_eye_rows_are_rewritten(saved_legacy_checkpoint, caplog):
    """The deploy branch's second review, point 7: the load's warning said every stored replay moment keeps
    the old form, which is false under F1 (the trainer rewrites the stored eye rows)."""
    path, _, _, _ = saved_legacy_checkpoint
    m = _lite(eye_row_convention="physical")
    with caplog.at_level("WARNING"):
        m.load_checkpoint(path, load_optimizer=False)
    text = " ".join(r.getMessage() for r in caplog.records if "differ from the checkpoint" in r.getMessage())
    assert "rewrites every stored ledger eye row" in text and "keep the old form until they age out" not in text


def test_switching_back_leaves_the_eye_entries_and_names_the_radius(saved_legacy_checkpoint, tmp_path):
    path, centre, _, _ = saved_legacy_checkpoint
    tmp = tmp_path / "order_avg_back"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        on = _lite(eye_row_convention="physical", executed_fields_only="on")
        on.load_checkpoint(path, load_optimizer=False)
        back_path = _save(on, tmp)
        back = _lite()
        back.load_checkpoint(back_path, load_optimizer=False)
        got = back.efference.order_centre
        assert torch.equal(got[EYE_XY], centre[EYE_XY]), "never negated, either way"
        assert float(got[RADIUS].abs().max()) == 0.0, "the free radii's mean is unknown: left at 0"
        assert back.motor_interface_change == {"eye_row_convention": ["physical", "legacy"],
                                               "executed_fields_only": [True, False]}
        assert any("switched off" in s for s in back.motor_interface_conversion["left_as_saved"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_only_an_average_this_load_restored_is_converted():
    m = _lite(eye_row_convention="physical", executed_fields_only="on")
    m.motor_interface_change = m._motor_interface_changes({})
    m.efference.order_centre.copy_(torch.arange(H * 9, dtype=torch.float32))
    m.efference.order_centre_weight = 0.3
    before = m.efference.order_centre.clone()
    # not restored by this load: left alone (it is not the checkpoint's)
    rec = m._convert_order_averages(efference_restored=False, self_state_loaded=False)
    assert torch.equal(m.efference.order_centre, before) and rec["averages"] == {}
    assert any("no saved average" in s for s in rec["left_as_saved"])
    # restored but cold (weight 0): the first lived moment sets it, already in the new form
    m.efference.order_centre_weight = 0.0
    m._convert_order_averages(efference_restored=True, self_state_loaded=False)
    assert torch.equal(m.efference.order_centre, before)


def test_the_grounded_self_calibration_gets_the_radius_conversion():
    centre = torch.linspace(-1.0, 1.0, H * 9)
    grounded = SimpleNamespace(action_center=centre.clone())
    stub = SimpleNamespace(motor_interface_change={"eye_row_convention": ["legacy", "physical"],
                                                   "executed_fields_only": [False, True]},
                           handle_names=NAMES, GAZE_IDX=GI, efference=None,
                           self_state=SimpleNamespace(grounded=grounded),
                           motor_interface_conversion=None)
    rec = LemLite._convert_order_averages(stub, efference_restored=False, self_state_loaded=True)
    want = centre.clone()
    want[RADIUS] = 0.0
    assert torch.equal(grounded.action_center, want)
    assert set(rec["averages"]) == {"self_state.grounded.action_center"}
    # not read by this load (the component was started fresh): left alone
    grounded.action_center = centre.clone()
    LemLite._convert_order_averages(stub, efference_restored=False, self_state_loaded=False)
    assert torch.equal(grounded.action_center, centre)


def test_the_saved_mean_moves_one_two_thousandth_a_moment_so_an_unconverted_offset_would_linger():
    """The arithmetic the conversion replaces: the running mean's own relaxation after a step change
    of its input (lem/model/efference.py update_centres) removes 1 - exp(-n/2000) of the offset in n
    lived moments; 18% by N + 400, when the deploy's kill lines have had their three windows."""
    m = _lite()
    lat = torch.zeros(1, m.n_mod, m.d_model)
    one = torch.ones(1, H, 9)
    m.efference.ensure_centres(lat, one)
    m.efference.order_centre_weight = 1.0            # a long-settled mean (weight at its limit)
    m.efference.order_centre.fill_(1.0)
    zero = torch.zeros(1, H, 9)
    for _ in range(400):
        m.efference.update_centres(lat, zero)
    removed = 1.0 - float(m.efference.order_centre[0])
    assert abs(removed - (1.0 - (1.0 - 1.0 / CENTRE_HORIZON_STEPS) ** 400)) < 1e-4
    assert 0.17 < removed < 0.19


# ------------------------------------------------------------------ the stored rows, rewritten
def _ledger(az, el, conv, B=2, seed=0):
    """A ledger order [B, H, 9] (cm) whose eye row is the row of the command (az, el) in `conv`."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(B, H, 9, generator=g).numpy() * 10.0
    return LemLite.gaze_to_handle(o, np.stack([az, el], axis=-1), convention=conv)


def _stored_learner():
    """A small learner with replay and the short histories filled with LEGACY ledger rows of known
    eye commands, as a companion saved before an F1 deploy restores them."""
    m = _lite(eye_row_convention="physical")
    m.enable_replay(2, size_per_env=16)
    rng = np.random.default_rng(3)
    az = rng.uniform(-0.6, 0.6, (2, 16))
    el = rng.uniform(-0.6, 0.4, (2, 16))
    for t in range(16):
        m.replay.executed[:, t] = torch.as_tensor(_ledger(az[:, t], el[:, t], "legacy", seed=t)).half()
    m.replay.has_executed.fill_(True)
    m.replay.count = m.replay.ptr = 16
    order = command_for_encoder(torch.as_tensor(_ledger(az[:, 0], el[:, 0], "legacy"), dtype=torch.float32))
    m._eff_hist.append({"latent": torch.zeros(1), "order": order.clone(), "step": 1})
    m._eff_tenure = {"order": order.clone(), "step": 1}
    m._eff_prev_order = torch.as_tensor(_ledger(az[:, 0], el[:, 0], "legacy"), dtype=torch.float32)
    m.motor_history = deque([order.clone(), order.clone()], maxlen=3)
    m.self_history = SimpleNamespace(events=deque([{"action": order.reshape(2, -1).clone()}]))
    m.self_rehearsal = SimpleNamespace(common=[{"actions": [order.reshape(2, -1).clone()]}],
                                       changed=[(0.5, {"actions": [order.reshape(2, -1).clone()]})])
    return m, az, el, order


def test_the_stored_ledger_rows_are_rewritten_as_the_new_conventions_rows_of_the_same_commands():
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, az, el, order = _stored_learner()
    ring_before = m.replay.executed.clone()
    prev = _ledger(az[:, 0], el[:, 0], "legacy")
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    prev2, done = switch_stored_eye_rows(m, prev)
    # the ring: exactly the physical rows of the commands each moment sent (f16 negation is exact)
    for t in range(16):
        want = torch.as_tensor(_ledger(az[:, t], el[:, t], "physical", seed=t)).half()
        assert torch.equal(m.replay.executed[:, t, GI], want[:, GI])
    other = torch.ones(H, dtype=torch.bool)
    other[GI] = False
    assert torch.equal(m.replay.executed[:, :, other], ring_before[:, :, other]), "nothing else changes"
    assert torch.equal(m.replay.executed[:, :, GI, [0, 1, 2, 5, 6, 7, 8]],
                       ring_before[:, :, GI, [0, 1, 2, 5, 6, 7, 8]])
    # the gaze copy and the gaze head read the elevation each stored moment SENT; unconverted, its mirror
    feats = gaze_order_features(command_for_encoder(m.replay.executed[:, 5].float()), GI, "physical")
    unconv = gaze_order_features(command_for_encoder(ring_before[:, 5].float()), GI, "physical")
    assert np.allclose(feats[:, 10].numpy(), el[:, 5], atol=2e-3)
    assert np.allclose(unconv[:, 10].numpy(), -el[:, 5], atol=2e-3)
    # the short histories and the trainer's last ledger, the same rewrite
    mirrored = LemLite.mirror_eye_rows(order, GI)
    assert torch.equal(m._eff_hist[0]["order"], mirrored) and torch.equal(m._eff_tenure["order"], mirrored)
    assert all(torch.equal(x, mirrored) for x in m.motor_history)
    assert torch.equal(m.self_history.events[0]["action"], mirrored.reshape(2, -1))
    assert torch.equal(m.self_rehearsal.common[0]["actions"][0], mirrored.reshape(2, -1))
    assert torch.equal(m.self_rehearsal.changed[0][1]["actions"][0], mirrored.reshape(2, -1))
    assert np.allclose(prev2, _ledger(az[:, 0], el[:, 0], "physical"), atol=1e-6)
    assert done["replay.executed"] == 32 and done["trainer.prev_executed"] == 1
    assert m.motor_interface_conversion["stored_eye_rows_rewritten"] == done
    # its own inverse: switching back restores every stored row exactly
    m.motor_interface_change = {"eye_row_convention": ["physical", "legacy"]}
    prev3, _ = switch_stored_eye_rows(m, prev2)
    assert torch.equal(m.replay.executed, ring_before) and np.allclose(prev3, prev, atol=0)


@pytest.mark.parametrize("change", [None, {"executed_fields_only": [False, True]}])
def test_without_an_eye_convention_switch_nothing_stored_is_touched(change):
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, az, el, order = _stored_learner()
    ring_before = m.replay.executed.clone()
    prev = _ledger(az[:, 0], el[:, 0], "legacy")
    m.motor_interface_change = change
    prev2, done = switch_stored_eye_rows(m, prev)
    assert done is None and prev2 is prev
    assert torch.equal(m.replay.executed, ring_before) and torch.equal(m._eff_hist[0]["order"], order)


def test_the_trainer_rewrites_the_stored_rows_after_the_companion_and_before_the_first_step():
    src = (PKG / "train_mimo.py").read_text(encoding="utf-8")
    i_restore = src.index("continuation_state.restore_learner(model,episode_resume['learner'])")
    i_switch = src.index("prev_executed, _eye_rows = switch_stored_eye_rows(model, prev_executed)")
    i_loop = src.index("while time.time() < deadline and (args.max_steps <= 0 or step < args.max_steps):")
    assert i_restore < i_switch < i_loop
    # the replay ring exists by then (the trainer enables it before it restores the companion)
    assert src.index("model.enable_replay(args.n_envs, size_per_env=args.replay_size)") < i_restore


# ------------------------------------------------------------------ the persistent memory's stored rows
def _memory_learner(steps=6, B=2, seed=1, spacing=0.3):
    """A small learner with the persistent memory whose history is written by the memory's OWN writer
    (memory_integration.observe) with LEGACY ledger rows of known eye commands. With 0.3 s between
    moments and the history's 0.5 s interval, six moments leave 3 retained events, a previous and a
    pending moment. Returns (model, [(az, el, ledger)], the last snapshot observe built)."""
    from mimo_bridge.memory_integration import observe
    m = _lite(eye_row_convention="physical", persistent_memory="on")
    rng = np.random.default_rng(seed)
    tokens = {mm: torch.randn(B, 1, m.d_model) for mm in m.active}
    mask = {mm: "real" for mm in m.active}
    cmds, snap = [], None
    for t in range(steps):
        az, el = rng.uniform(-0.5, 0.5, B), rng.uniform(-0.5, 0.4, B)
        ex = torch.as_tensor(_ledger(az, el, "legacy", B=B, seed=t), dtype=torch.float32)
        frames = [SimpleNamespace(sim_time=spacing * t) for _ in range(B)]
        _, _, snap, _ = observe(m, tokens, mask, {}, frames, ex)
        cmds.append((az, el, ex))
    return m, cmds, snap


def test_a_memory_events_order_columns_hold_the_ledger_row_and_have_not_moved():
    """memory_integration.observe writes the executed order (command_for_encoder of the ledger row) at
    action_columns, and the eye row's x and y at eye_row_columns. PINNED: those columns start after one
    d_model token per active sense and the ten need columns, and the event is as wide as the memory
    reads. If a memory event's layout changes, this fails, and LemLite.convert_stored_eye_rows (which
    rewrites those columns at an eye row convention switch) must be checked against the new layout."""
    from mimo_bridge.memory_integration import action_columns, eye_row_columns
    m, cmds, snap = _memory_learner(steps=2)
    a = action_columns(m)
    assert a.start == len(m.active) * m.d_model + 10 and a.stop - a.start == H * 9
    assert snap["events"].shape[-1] == m.persistent_memory.input_size == a.stop + m.d_model
    assert eye_row_columns(m) == [a.start + GI * 9 + 3, a.start + GI * 9 + 4]
    assert torch.equal(m.memory_history.events[0][:, a], command_for_encoder(cmds[0][2]).reshape(2, -1))
    assert torch.equal(m.memory_history.previous[:, a], command_for_encoder(cmds[1][2]).reshape(2, -1))
    assert torch.equal(snap["events"][:, -1, a], command_for_encoder(cmds[1][2]).reshape(2, -1))


def _mirror_cols(x, cols):
    out = x.clone()
    out[..., cols] = -out[..., cols]
    return out


@pytest.mark.parametrize("shared", [True, False])
def test_the_persistent_memorys_stored_rows_are_rewritten_through_its_own_writer(shared):
    """The deploy branch's second review, finding 1. Reproduced before this rewrite existed (its probe
    probe_memory_rows.py): after the switch, 6 of 6 memory events and the memory's previous moment kept
    the legacy rows while the trainer's ledger was rewritten. Now every stored copy - the retained
    events, the previous and the pending moment, the delayed lesson's snapshot and the tenure's - holds
    the new convention's row of the command it recorded, every other column bit for bit, a snapshot
    held under both names rewritten once; the next observe continues on the rewritten history."""
    from mimo_bridge.memory_integration import action_columns, eye_row_columns, observe
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, cmds, snap = _memory_learner()
    mh = m.memory_history
    assert len(mh.events) == 3 and mh.previous is not None and mh.pending is not None
    older = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in snap.items()}
    m._nf_memory = snap
    m._tenure_pending = {"memory": snap if shared else older, "order": torch.zeros(2, H, 9)}
    cols = eye_row_columns(m)
    events0 = [e.clone() for e in mh.events]
    prev0, pend0 = mh.previous.clone(), mh.pending.clone()
    snap0, older0 = snap["events"].clone(), older["events"].clone()
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    _, done = switch_stored_eye_rows(m, cmds[-1][2].numpy())
    for after, before in zip(mh.events, events0):
        assert torch.equal(after, _mirror_cols(before, cols))
    assert torch.equal(mh.previous, _mirror_cols(prev0, cols))
    assert torch.equal(mh.pending, _mirror_cols(pend0, cols))
    assert torch.equal(m._nf_memory["events"], _mirror_cols(snap0, cols))
    assert torch.equal(m._tenure_pending["memory"]["events"], _mirror_cols(snap0 if shared else older0, cols))
    # ... which are the rows the physical convention writes for the same commands
    az, el, _ = cmds[-1]
    phys = command_for_encoder(torch.as_tensor(_ledger(az, el, "physical", B=2, seed=len(cmds) - 1),
                                               dtype=torch.float32))
    assert torch.allclose(mh.previous[:, action_columns(m)], phys.reshape(2, -1), atol=1e-6)
    # the record: every store with what reads it, and the encoded carriers no rewrite reaches
    assert done["persistent_memory.history"] == 5
    assert done["persistent_memory._nf_memory"] == done["persistent_memory._tenure_pending"] == snap0.shape[1]
    conv = m.motor_interface_conversion
    assert all(conv["stored_eye_rows_readers"][k] != "unlisted" for k in done)
    assert {"replay.memory_states", "persistent_memory.carry", "context_buffer", "imagination.ctx_history",
            "imagination.memory_bias_history"} <= set(conv["encoded_eye_rows_not_rewritten"])
    # the third review, point G: every encoded carrier the episode companion restores is named (the
    # imagination chain's histories of encoded contexts and memory biases were missing)
    from mimo_bridge.continuation_state import DATA_NAMES
    assert {"ctx_history", "memory_bias_history"} <= DATA_NAMES
    # the memory's own writer goes on from the rewritten history with the new convention's rows
    az2, el2 = np.array([0.2, -0.3]), np.array([0.1, -0.2])
    ex2 = torch.as_tensor(_ledger(az2, el2, "physical", B=2, seed=99), dtype=torch.float32)
    tokens = {mm: torch.randn(2, 1, m.d_model) for mm in m.active}
    _, _, snap2, _ = observe(m, tokens, {mm: "real" for mm in m.active}, {},
                             [SimpleNamespace(sim_time=0.3 * len(cmds)) for _ in range(2)], ex2)
    assert torch.equal(snap2["events"][:, 0], _mirror_cols(events0[0], cols))
    assert torch.equal(snap2["events"][:, -1, action_columns(m)], command_for_encoder(ex2).reshape(2, -1))


def _repaired_memory_available():
    from lem.model.persistent_memory import PersistentMemory
    return hasattr(PersistentMemory, "enable_repair")


def _recorded_ring(m, B=2, size=200, seed=4, world_el=0.12):
    """The relaunched learner's replay ring as the companion restores it: LEGACY ledger rows, each the
    model's gaze command plus a world part of mean elevation world_el, with the command recorded beside
    it (ReplayMemory.ctrl_gaze)."""
    m.enable_replay(B, size_per_env=size)
    rng = np.random.default_rng(seed)
    az_c, el_c = rng.uniform(-0.4, 0.4, (B, size)), rng.uniform(-0.5, 0.3, (B, size))
    w_el, w_az = world_el + rng.normal(0, 0.05, (B, size)), rng.normal(0, 0.05, (B, size))
    for t in range(size):
        m.replay.executed[:, t] = torch.as_tensor(
            _ledger(az_c[:, t] + w_az[:, t], el_c[:, t] + w_el[:, t], "legacy", B=B, seed=t)).half()
        m.replay.ctrl_gaze[:, t] = torch.as_tensor(np.stack([az_c[:, t], el_c[:, t]], -1), dtype=torch.float32)
    m.replay.has_executed.fill_(True)
    m.replay.has_ctrl_gaze.fill_(True)
    m.replay.count = m.replay.ptr = size


def event_centre_on_a_switched_checkpoint(tmp, record=True):
    """The memory repair's running mean of events (fix/persistent-memory-2026-09-25) at an F1 switch,
    through the repair's own code: a legacy run whose memory the repair switched at its first lived moment
    (memory_integration.observe calls the switch) and whose running mean then followed 11 more moments,
    saved; a 'physical' relaunch with the same repair loads it, the episode companion's history and a
    replay ring are put back, and the trainer's switch runs. Returns (the centre as saved, the relaunched
    learner, the record). By the order centre's method (the coordinator's decision after the four F1
    follow-ups: one method for both centres), which since the fourth review's fix round leaves the eye
    columns as saved, with the reason, whether the ring holds recorded commands (record=True) or not."""
    from mimo_bridge.memory_integration import eye_row_columns, observe
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    B = 2
    src = _lite(eye_row_convention="legacy", persistent_memory="on")
    src.persistent_memory.enable_repair("centred")
    if not hasattr(src, "_persistent_memory_switch"):
        # the repair branch's learner method folds the optimizer and the ring too; the memory's own switch
        # is what sets the centre
        src._persistent_memory_switch = lambda snap: src.persistent_memory.switch(snap)
    rng = np.random.default_rng(5)
    tokens = {mm: torch.randn(B, 1, src.d_model) for mm in src.active}
    mask = {mm: "real" for mm in src.active}
    for t in range(12):
        # elevations mostly below level: the eye columns' running mean is not 0
        az, el = rng.uniform(-0.4, 0.4, B), rng.uniform(-0.6, 0.1, B)
        ex = torch.as_tensor(_ledger(az, el, "legacy", B=B, seed=t), dtype=torch.float32)
        observe(src, tokens, mask, {}, [SimpleNamespace(sim_time=0.3 * t) for _ in range(B)], ex)
    pm = src.persistent_memory
    assert pm.repair_switched_py and float(pm.event_centre_weight) > 0.0
    path = _save(src, tmp)
    m = _lite(eye_row_convention="physical", persistent_memory="on")
    m.persistent_memory.enable_repair("centred")
    m.load_checkpoint(path, load_optimizer=False)
    assert m.persistent_memory.repair_switched_py, "the switched checkpoint loads switched"
    m.memory_history.load_state_dict(src.memory_history.state_dict(), m.device)     # the companion's history
    _recorded_ring(m, B=B)
    if not record:
        m.replay.has_ctrl_gaze.fill_(False)
    saved = m.persistent_memory.event_centre.detach().clone()
    assert torch.equal(saved, pm.event_centre)
    m.motor_interface_conversion = dict(m.motor_interface_conversion or {})
    _, done = switch_stored_eye_rows(m, None)
    cols = eye_row_columns(m)
    rec = m.motor_interface_conversion["event_centre_eye_shift"]
    got = m.persistent_memory.event_centre
    assert torch.equal(got, saved) and rec["shift"] is None, "left as saved, with the ring's commands or without"
    assert rec["left_as_saved"] == LemLite.CENTRE_LEFT_AS_SAVED and rec["entries"] == [int(c) for c in cols]
    assert "persistent_memory.event_centre" not in done
    # one method for both centres: the order centre's record says the same
    oc = m.motor_interface_conversion.get("order_centre_eye_shift")
    assert oc is None or (oc["shift"] is None and oc["left_as_saved"] == rec["left_as_saved"])
    assert float(saved[cols].abs().sum()) > 0.0
    return saved, m, done


@pytest.mark.skipif(not _repaired_memory_available(),
                    reason="the persistent memory repair (fix/persistent-memory-2026-09-25) is not merged here")
@pytest.mark.parametrize("record", [True, False])
def test_the_repaired_memorys_event_centre_is_left_as_saved_on_a_switched_checkpoint(record, tmp_path):
    tmp = tmp_path / "event_centre"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        event_centre_on_a_switched_checkpoint(tmp, record=record)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_one_method_leaves_both_centres_as_saved():
    """Without the repair's code: a stand-in running mean of events (set) and the order centre, at the same
    switch, are both left as saved with the same reason, although the ring holds recorded commands (the fourth
    review's fix round: the ring-measured move is gone); the stored events are rewritten as before."""
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, cmds, _ = _memory_learner()
    _recorded_ring(m, B=2)
    width = int(m.persistent_memory.input_size)
    centre = torch.linspace(-1.0, 1.0, width)
    m.persistent_memory.register_buffer("event_centre", centre.clone())
    m.persistent_memory.register_buffer("event_centre_weight", torch.ones((), dtype=torch.float64))
    oc = torch.linspace(-0.5, 0.5, H * 9)
    m.efference.order_centre.copy_(oc)
    m.efference.order_centre_weight = 1.0
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    _, done = switch_stored_eye_rows(m, None)
    ev = m.motor_interface_conversion["event_centre_eye_shift"]
    od = m.motor_interface_conversion["order_centre_eye_shift"]
    assert ev["shift"] is None and od["shift"] is None and ev["left_as_saved"] == od["left_as_saved"]
    assert torch.equal(m.persistent_memory.event_centre, centre) and torch.equal(m.efference.order_centre, oc)
    assert "persistent_memory.event_centre" not in done and done["persistent_memory.history"] == 5


def test_without_the_repaired_memory_no_centre_is_touched():
    """Absent (a memory without the repair, or a repair whose centre was never set): nothing."""
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m, cmds, _ = _memory_learner()
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    _, done = switch_stored_eye_rows(m, None)
    assert "persistent_memory.event_centre" not in done and done["persistent_memory.history"] == 5
    # a stand-in memory that carries a centre never set (weight 0) is left as it is
    centre = torch.linspace(-1.0, 1.0, m.persistent_memory.input_size)
    m.persistent_memory.register_buffer("event_centre", centre.clone())
    m.persistent_memory.register_buffer("event_centre_weight", torch.zeros((), dtype=torch.float64))
    _, done = switch_stored_eye_rows(m, None)
    assert "persistent_memory.event_centre" not in done
    assert torch.equal(m.persistent_memory.event_centre, centre)


def test_the_switch_refuses_a_memory_event_of_another_width_before_rewriting_anything():
    m, cmds, snap = _memory_learner(steps=2)
    m.enable_replay(2, size_per_env=4)
    m.replay.executed.copy_(torch.randn(m.replay.executed.shape).half())
    ring = m.replay.executed.clone()
    m.memory_history.events[0] = torch.zeros(2, 7)
    with pytest.raises(ValueError, match="eye row columns are not known"):
        m.convert_stored_eye_rows()
    assert torch.equal(m.replay.executed, ring), "nothing is rewritten when the memory's layout is unknown"
