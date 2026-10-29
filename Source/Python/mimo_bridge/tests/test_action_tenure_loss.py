# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The action-tenure loss: the order's marginal contribution, graded at the order's END.

Why it exists, measured: the model learned and unlearned its action across consecutive
checkpoints for a week (H41/H42 flicker 0-400%; H44 held five hours; H45 gave it all
back, both instruments agreeing), because the base objective can be minimised while
ignoring the command. And the graded horizon was one 50 ms step - ~1 cm of movement -
while an order's tenure is 250 ms, where it visibly bends the trajectory.

The loss: when an order ends, predict the tenure's change from the state where it began,
once WITH the order and once with "do nothing" (stop-gradient baseline off the same
weights), and hinge the with-order branch to beat the baseline by alpha. These tests pin
the flag default, the checkpoint registration (the recon_decoders trap), the stop-grad,
the reset discipline, and that a live run produces the fields.
"""
import inspect

import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import LemLite
from mimo_bridge import train_mimo as TM
from mimo_bridge import lem_model as LM


def _model(on, **kw):
    kw.setdefault("next_frame_loss", True)      # the tenure lesson rides the nf caches
    if kw.get("agency_drive"):
        kw.setdefault("intention", False) or kw.update(intention=True)
    kw.setdefault("intention", False)
    return LemLite(device="cpu", seed=0, imagination=False,
                   action_tenure_loss=on, **kw)


# ------------------------------------------------------------------ default and wiring
def test_off_by_default_and_no_heads_when_off():
    m = _model(False)
    assert m.action_tenure is False
    assert m.tenure_heads is None
    assert "tenure_heads" not in m._side_modules()


def test_on_builds_heads_and_registers_them_for_checkpoints():
    """recon_decoders and td_gates were silently never saved for weeks; any new side
    module must be in _side_modules() or every resume rebuilds it from random init."""
    m = _model(True)
    assert m.tenure_heads is not None
    assert set(m.tenure_heads.keys()) == {"pose"}, (
        "re-targeted 2026-09-01: the head predicts the achieved-pose change; the "
        "order's token-space share measured ~0 while its physical share is +4.4 pts")
    assert m._side_modules()["tenure_heads"] is m.tenure_heads


def test_the_trainer_exposes_the_flag_and_plumbs_it():
    src = inspect.getsource(TM)
    assert '"--action-tenure-loss"' in src
    assert 'choices=["on", "off"], default="off"' in src.split('"--action-tenure-loss"')[1][:200]
    assert 'action_tenure_loss=args.action_tenure_loss == "on"' in src
    assert 'default=0.04' in src.split('--action-tenure-alpha')[1][:120], (
        'trainer and constructor defaults must agree (review R11)')
    assert '"tenure_pe"' in src and '"tenure_gap"' in src


# ------------------------------------------------------------------ the safety points
def test_the_null_branch_is_stop_gradient():
    """The hinge must be satisfiable only by reading the order - never by making the
    do-nothing baseline worse. Source-level: the null branch runs under no_grad and its
    error is detached before entering the hinge."""
    src = inspect.getsource(LM.LemLite.step)
    assert "pred_null = _tenure_forward(row0, grad=False)" in src
    assert ".detach()   # STOP-GRAD baseline" in src
    assert "mw / m0" in src, (
        "the hinge must be SCALE-FREE: the absolute form shrank with the error "
        "pie to ~0.002% of total loss on H47 and taught nothing")
    # review M2: below the measured still-body pose noise (limp crib body, 5-step
    # displacement mse p90 4.3e-7) the ratio is noise-over-noise with d/dmw = 1/m0;
    # the hinge is skipped there instead of clamped to a 1e-10 denominator
    assert "self._tenure_m0_floor" in src
    assert "m0.clamp_min(1e-10)" not in src


def test_do_nothing_is_the_identity_rotation_not_the_zero_quaternion():
    """torch.zeros_like alone gives an all-zero quaternion - not a rotation at all,
    off-distribution rather than uninformative (the causal test learned this first)."""
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("ACTION-TENURE LESSON")
    assert "no[..., 6] = 1.0" in src[i:i + 4000]


def test_reset_clears_the_pending_lesson():
    m = _model(True)
    m._tenure_pending = {"tokens": {}, "ctx": None, "ach": None, "order": torch.zeros(1, 24, 9)}
    m._tenure_prev_order = torch.zeros(1, 24, 9)
    m._tenure_bank.append((torch.zeros(3), torch.zeros(3)))
    m.reset_transition_state()
    assert m._tenure_pending is None and m._tenure_prev_order is None
    assert len(m._tenure_bank) == 0, "stale decoys would cross a teleport"


# ------------------------------------------------------------------ it actually runs
def test_a_live_run_fires_the_lesson_and_logs_the_fields(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False,
                next_frame_loss=True, action_tenure_loss=True,
                overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "tn"), tag="_tn",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        fired = saw_term = False
        for i in range(26):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            assert res.loss == res.loss, "loss went NaN with the tenure loss on"
            if res.tenure_pe_per_modality:
                fired = True
            if "tenure" in res.breakdown:   # present only on the step an order ends
                saw_term = True
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert fired, "26 steps with a 5-step hold never completed a tenure"
        assert res.tenure_gap_per_modality, "the gap - the order's worth - was not logged"
        assert saw_term, "the loss term never reached the total on any firing step"
    finally:
        src.close()


def test_with_the_flag_off_nothing_changes(tmp_path):
    """No fields, no loss term, no heads - a run without the flag is the old run."""
    from mimo_bridge.datasource import SingleEnvSource
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False,
                next_frame_loss=True, action_tenure_loss=False,
                overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "tf"), tag="_tf",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(12):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert not res.tenure_pe_per_modality
        assert "tenure" not in res.breakdown
    finally:
        src.close()


# ------------------------------------------------- token-reach: gradient to the ENCODERS
def test_token_reach_off_by_default_and_heads_only_when_on():
    m = _model(True)
    assert m.tenure_token_reach is False
    assert not any(k.startswith("tok_") for k in m.tenure_heads.keys())
    m2 = _model(True, action_tenure_token_reach=True)
    assert m2.tenure_token_reach is True
    assert "order_proj" in m2.tenure_heads, (
        "the encoder branch is order-to-outcome matching (bench-validated); the "
        "hinge/plain forms collapsed token variance live (H49, H50) and on the bed")
    assert m2._side_modules()["tenure_heads"] is m2.tenure_heads


def test_every_side_module_parameter_is_optimized():
    """The checkpoint trap has an optimizer cousin: tenure_heads were saved and loaded
    but NEVER in the optimizer for the first five tenure runs - the pose head and the
    order projection sat frozen at random init, the trunk learned around the frozen
    readout, and the matching term was impossible (a frozen random projection defines
    an arbitrary target). Generic guard: every parameter of every side module must be
    in the optimizer's param groups."""
    m = _model(True, action_tenure_token_reach=True)
    opt = {id(p) for g in m.optimizer.param_groups for p in g["params"]}
    for name, mod in m._side_modules().items():
        if mod is None:
            continue
        ps = list(mod.parameters())
        missing = [i for i, p in enumerate(ps) if id(p) not in opt]
        assert not missing, f"side module {name}: {len(missing)}/{len(ps)} params not optimized"


def test_token_branch_is_order_to_outcome_matching():
    """Pins the bench-validated form. Both hinge variants collapsed token variance ~10x
    (H49 and H50 live; reproduced on the bed, vision 0.24 -> 0.01), because their
    gradient always offers variance-shrink as a descent direction. Matching scores are
    relative - shrinking raises no answer above the others - and the bed shows
    discrimination 0.00 -> 1.00 in 300 encoder steps with variance held. Both ends of
    the change are re-encoded live so the gradient reaches the encoders; decoys come
    from a detached ring that clears on reset."""
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("Both ends of the change re-encoded LIVE")
    # marker-bounded, not byte-counted (2026-09-08: two growth-triggered false
    # failures in one day): the block runs to its own last asserted statement
    j = src.index("self._tenure_bank.append(delta.detach())", i)
    blk = src[i:j + 200]
    # review M14: encode() is per-modality, so the old motor-row swap was dead weight;
    # the end frame is re-encoded senses-only and the order reaches the loss through
    # order_proj alone
    assert "te_l = _live_tokens(self._nf_raw, self._nf_mask)" in blk
    assert 'self.tenure_heads["order_proj"]' in blk
    assert "F.cross_entropy" in blk
    # review E22: the bank stores only the delta (q was stored and never read)
    assert "self._tenure_bank.append(delta.detach())" in blk
    # HAND-REGARD GATE: vision's slice participates only when the hand is in the field
    # of view at both ends of the tenure. Demanding visual command-traces from frames
    # that contain none eroded vision's causal error ~5x over six H55 hours; removing
    # vision entirely would abandon visual self-recognition, which the project exists
    # to build. The gate is the infant's curriculum: learn it while watching the hand.
    assert "HAND-REGARD GATE" in blk
    # the env flag is READ outside the matching block (where _nf_handvis is
    # cached), so it is asserted on the whole step source, not the block
    assert 'hand_in_view' in src
    assert "dpart = dpart * hv" in blk
    # SLICE NORMALIZATION: H54's matching crushed vision's tenure-change 5x by
    # silencing the near-zero-signal slice; per-body unit-norm makes suppression
    # worthless. CROSS-BODY DECOYS: the other bodies' same-instant outcomes join the
    # lineup - decoys with zero staleness.
    assert "dpart = dpart / (dpart.std(dim=-1, keepdim=True) + 1e-6)" in blk
    assert "torch.roll(delta.detach(), shift, dims=0)" in blk


def test_token_reach_fires_and_reaches_the_vision_encoder(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False,
                next_frame_loss=True, action_tenure_loss=True,
                action_tenure_token_reach=True,
                overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "tr"), tag="_tr",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        fired = saw_tok = False
        for i in range(26):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            assert res.loss == res.loss
            if res.tenure_tok_per_modality:
                fired = True
            if "tenure_tok" in res.breakdown:  # per-step; the logged fields persist
                saw_tok = True
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert fired, "the token-reach term never fired in 26 steps"
        assert saw_tok, "the token-reach term never reached the loss total"
    finally:
        src.close()


# --------------------------------------------------------------- the agency drive
def test_agency_drive_off_by_default_and_requires_match():
    m = _model(True, action_tenure_token_reach=True)
    assert m.agency_drive is False
    m2 = _model(True, action_tenure_token_reach=True, agency_drive=True)
    assert m2.agency_drive is True
    with pytest.raises(ValueError, match="requires the order-to-outcome matching"):
        _model(True, agency_drive=True)    # no match term -> no evidence -> refuse loudly


def test_displacement_is_touch_parity_not_a_clock():
    """The bootstrap fades as vision's live agency evidence reaches touch's - both
    measured EMAs - never on exposure or time. Wrong guesses fail safe in both
    directions: weak vision keeps the prior (keep practicing), early skill retires it."""
    m = _model(True, action_tenure_token_reach=True, agency_drive=True, intention=True)
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("hand-watching bootstrap")
    blk = src[max(0, i - 2500):i + 3000]
    assert "self._vis_adv_ema" in blk and "self._touch_adv_ema" in blk
    assert "displacement, not a clock" in blk
    assert "min(1.0, max(0.0" in blk, "the prior must be budget-clamped both ways"
    m._vis_adv_ema, m._touch_adv_ema = 0.10, 0.10
    assert max(0.0, 1.0 - m._vis_adv_ema / max(m._touch_adv_ema, 1e-6)) == 0.0
    m._vis_adv_ema = 0.0
    assert max(0.0, 1.0 - m._vis_adv_ema / max(m._touch_adv_ema, 1e-6)) == 1.0


def test_the_bootstrap_pays_only_for_a_moving_hand():
    """A still-hand stare must earn nothing - the degenerate equilibrium the red-team
    found (park gaze on a motionless hand, get paid forever, learn nothing)."""
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("hand-watching bootstrap")
    blk = src[max(0, i - 500):i + 3000]
    assert "w_prior * motion * fov" in blk, "prior must be gated by hand MOTION and fovea"


def test_agency_payment_is_bounded_rank_not_raw_gap():
    """Raw gap magnitude rewards thrashing (idiosyncratic flailing separates from
    decoys); the payment is the bounded rank outcome. Measured before building:
    corr(bounded payment, order magnitude) = -0.09."""
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("AGENCY DRIVE" if "AGENCY DRIVE" in src else "agency_drive:")
    assert "pay = (torch.argmax(logits, dim=1) == 0).float()" in src
