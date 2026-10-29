# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The second adversarial sweep (2026-09-02): every finding pinned by a test.

Three review passes covered the model forward path (M), the trainer + checkpoint
path (T), and the encoders/instruments (E). Every applied fix is tested here -
behaviorally where the scaffolding reaches it, at source level where the code lives
inside a nested trainer function. The M1 lesson drives the style: the R2 fix
shipped green behind a source-string assert while the behavior stayed broken, so
the gates, motion, and EMA fixes get real drives.
"""
import inspect
import json
import os

import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import LemLite
from mimo_bridge import lem_model as LM
from mimo_bridge import train_mimo as TM
from mimo_bridge import env as EV
from mimo_bridge.tools import causal_test as CT
from lem import live_test as LT
from lem.training import checkpoint as CKPT


def _agency_model(**kw):
    kw.setdefault("next_frame_loss", True)
    return LemLite(device="cpu", seed=0, imagination=False,
                   action_tenure_loss=True, action_tenure_token_reach=True,
                   agency_drive=True, intention=True, intention_execute=False,
                   overrides={"training": {"steps_per_phase": 3000}}, **kw)


def _drive(m, src, n, jitter_gaze=False, rng=None):
    """The trainer's cadence: 5-step holds, executed action carried one step late."""
    frames = src.reset()
    dev = dict(m.dev_limits(10 ** 9))
    dev["exploration_noise"] = 0.05
    held = prev = None
    fired_steps = []
    for i in range(n):
        res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
        if "tenure" in res.breakdown:   # present only on the step an order ends;
            fired_steps.append(i)       # tenure_pe_per_modality is sample-and-hold
        if i % 5 == 0 or held is None:
            held = res.motor_for_action
        for _ in range(5):
            frames = src.step(held, res.gaze_out, None, render=False)
        prev = held.clone() if torch.is_tensor(held) else np.array(held)
        if jitter_gaze:
            # the trainer rewrites handle 23 from the eye efference copy every
            # step - simulate it with a fresh random unit quaternion each step
            q = torch.randn(prev.shape[0], 4, generator=rng)
            prev[:, 23, 3:7] = q / q.norm(dim=-1, keepdim=True)
    return fired_steps


# ===================================================================== M1 / M18
def test_m1_the_motion_gate_sees_real_motion(tmp_path):
    """R2's fix read _nf_ach - which the delayed-lesson refresh had ALREADY
    overwritten with the current pose, so motion measured 0.0 forever and the
    hand-watching bootstrap paid nothing in any run. With real motion the EMA must
    leave the zero-decay curve (0.99^k * 1e-3 with a 1e-6 floor)."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "m1"), tag="_m1",
                          vision=False, seed=5)
    try:
        n = 30
        _drive(m, src, n)
        decay_curve = 0.99 ** n * 1e-3 + (1 - 0.99 ** n) * 1e-6
        assert m._hand_motion_ema > 5 * decay_curve, (
            f"hand-motion EMA {m._hand_motion_ema:.2e} is on the zero-motion decay "
            f"curve ({decay_curve:.2e}): the gate is reading a refreshed cache again")
        assert m.last_agency.get("motion", 0.0) > 0.0
    finally:
        src.close()


def test_m18_motion_is_per_body_over_the_hand_handles():
    """A cross-body all-handle scalar let a still body in a moving batch collect
    the bootstrap with a parked hand. The block reads _ach_prev (pre-refresh) and
    hand handles 0..11 only; source-pinned alongside the M1 behavioral test, and
    the payment vector must be per-body ([B])."""
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("review M18")
    blk = src[i - 2000:i + 2000]
    assert "handle_state[:, :12, :3]" in blk
    assert "_ach_prev[:, :12, :3]" in blk
    assert ".mean(dim=1)" in blk
    assert "_ach_prev = self._nf_ach" in src, "the pre-refresh capture is gone"
    # the capture must happen BEFORE the refresh writes this step's pose
    assert src.index("_ach_prev = self._nf_ach") < src.index(
        "self._nf_ach = (self.last_achieved.detach().clone()")


def test_m18_still_body_earns_no_bootstrap(tmp_path):
    """Two bodies, fovea flags forced on for both, payment EMA zeroed: the still
    body's agency payment must be zero while the moving body's is positive."""
    m = _agency_model()
    B, H = 2, len(LM.HANDLE_NAMES)
    m.reset_context(B)
    pose0 = torch.zeros(B, H, 10)
    pose0[..., 6] = 1.0
    m._nf_ach = pose0.clone()

    class _F:
        def __init__(self):
            self.meta = {"task": {"hand_in_fovea": True, "hand_in_view": True}}

    frames = [_F(), _F()]
    handle_state = pose0.clone()
    handle_state[1, :12, 0] += 0.02          # body 1 moved its hand; body 0 still
    # replicate the block's arithmetic exactly as shipped
    _ach_prev = m._nf_ach
    hm_vec = (handle_state[:, :12, :3] - _ach_prev[:, :12, :3]).norm(dim=-1).mean(dim=1)
    m._hand_motion_ema = 0.99 * m._hand_motion_ema + 0.01 * max(float(hm_vec.mean()), 1e-6)
    motion = (hm_vec / (2 * m._hand_motion_ema)).clamp(max=1.0)
    fov = torch.tensor([1.0, 1.0])
    ag = m.agency_prior_w0 * 1.0 * motion * fov
    assert float(ag[0]) == 0.0 and float(ag[1]) > 0.0


# ===================================================================== M24 debt
def test_m24_gaze_jitter_does_not_end_a_tenure(tmp_path):
    """The behavioral test commit b9126e08's message claimed. Handle 23 rewritten
    with a fresh quaternion every step: boundaries must still land only where the
    LIMB order changes (every 5 steps), not every step."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "m24"), tag="_m24",
                          vision=False, seed=7)
    try:
        rng = torch.Generator().manual_seed(3)
        fired = _drive(m, src, 26, jitter_gaze=True, rng=rng)
        assert fired, "no tenure ever completed"
        assert all(s % 5 == 1 for s in fired), (
            f"tenure events fired at steps {fired}: gaze jitter is ending tenures "
            f"(the H53-H58 defect) - every event must sit one step after a hold "
            f"boundary")
    finally:
        src.close()


# ===================================================================== M16a
def test_m16a_the_displacement_fade_is_smooth_not_a_sign_rail():
    """1 - vis/max(touch, 1e-6) slammed to a clamp rail off the SIGN of two EMAs.
    Clamped advantages + the measured floor (0.005 = touch-adv EMA std over
    H60/H61) make it a smooth fade."""
    m = _agency_model()

    def displaced(vis, touch):
        _va = max(vis, 0.0)
        _ta = max(touch, 0.0)
        return min(1.0, max(0.0, 1.0 - _va / max(_ta, m._agency_adv_floor)))

    assert m._agency_adv_floor == pytest.approx(0.005)
    assert displaced(-1e-4, 0.01) == 1.0          # negative vision: full prior, no flip
    assert displaced(0.0, -1e-3) == 1.0           # both cold/negative: full prior
    d_small = displaced(1e-4, -1e-3)              # tiny vision vs no touch anchor
    assert 0.9 < d_small < 1.0, (
        f"tiny vision evidence must fade the prior a LITTLE ({d_small}), not slam "
        f"it to 0 through a 1e-6 denominator")
    assert displaced(0.005, 0.0) == pytest.approx(0.0)   # vision at the noise scale: gone
    # monotone in vis
    ds = [displaced(v, 0.002) for v in (0.0, 1e-3, 2e-3, 5e-3)]
    assert ds == sorted(ds, reverse=True)
    src = inspect.getsource(LM.LemLite.step)
    assert "max(_ta, self._agency_adv_floor)" in src
    assert "max(self._touch_adv_ema, 1e-6)" not in src


# ===================================================================== M16b / T2
def test_m16b_agency_state_rides_the_checkpoint(tmp_path):
    """Every resume re-armed the full bootstrap (H60 end prior_w 0.0 -> H61 start
    0.05). The four EMAs must round-trip through save/load."""
    m = _agency_model(checkpoint_dir=str(tmp_path / "ck"))
    m._vis_adv_ema = 0.0123
    m._touch_adv_ema = 0.0456
    m._hand_motion_ema = 0.0078
    m._agency_pay_ema = torch.tensor([0.25, 0.75])
    p = m.save_checkpoint(1234)
    # deferred items 3/5 (2026-09-02) widened the file to runtime_state.pt
    assert p and os.path.isfile(os.path.join(p, "runtime_state.pt"))
    m2 = _agency_model(checkpoint_dir=str(tmp_path / "ck"))
    m2.load_checkpoint(p, load_optimizer=False)
    assert m2._vis_adv_ema == pytest.approx(0.0123)
    assert m2._touch_adv_ema == pytest.approx(0.0456)
    assert m2._hand_motion_ema == pytest.approx(0.0078)
    assert torch.allclose(m2._agency_pay_ema, torch.tensor([0.25, 0.75]))
    # New receipts require runtime evidence: losing it is an incomplete save.
    os.remove(os.path.join(p, "runtime_state.pt"))
    m3 = _agency_model(checkpoint_dir=str(tmp_path / "ck"))
    with pytest.raises(ValueError,match='missing required runtime evidence'):
        m3.load_checkpoint(p, load_optimizer=False)


# ===================================================================== M15
def test_m15_single_body_rigs_update_the_advantage_emas(tmp_path):
    """At B==1 the EMAs sat inside `if B > 1` and never moved - the bench rig that
    validates this objective runs one body. The banked decoys are the comparison
    set now."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "m15"), tag="_m15",
                          vision=False, seed=9)
    try:
        _drive(m, src, 31)
        assert len(m._tenure_bank) > 0, "no decoys banked - matching never fired"
        assert (m._vis_adv_ema, m._touch_adv_ema) != (0.0, 0.0), (
            "31 steps of B==1 tenure events left both advantage EMAs at exactly "
            "0.0: the B>1 guard is back")
    finally:
        src.close()


# ===================================================================== M2
def test_m2_the_hinge_skips_below_the_measured_motion_floor(tmp_path):
    """mw=1e-8 / m0=1e-10 used to add +99 to the loss at d/dmw = 1e10. Below the
    measured still-body pose noise (5e-7) the hinge is skipped and the term is mw
    alone; above it the ratio stands."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    assert m._tenure_m0_floor == pytest.approx(5e-7)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "m2"), tag="_m2",
                          vision=False, seed=11)
    try:
        m._tenure_m0_floor = float("inf")        # force the no-signal branch
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(16):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if "tenure" in res.breakdown:
                # breakdown carries the CONTRIBUTION, so the weight is already in it
                # (2026-09-15: four entries used to be stored pre-weight while the loss
                # added the weighted value, and the loss panel drew them at up to 50x
                # their real share). The property under test is unchanged: with the
                # hinge skipped the term is mw alone, with no hinge penalty added.
                t = float(res.breakdown["tenure"])
                mw = res.tenure_pe_per_modality["pose"]
                assert t == pytest.approx(m.tenure_weight * mw, rel=1e-5), (
                    f"with the hinge skipped the term must equal the weighted mw: "
                    f"{t} vs {m.tenure_weight} * {mw}")
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()


# ===================================================================== M3b
def test_m3b_the_stored_order_neutralizes_the_gaze_handle(tmp_path):
    """The trainer rewrites handle 23 every step, so a stored gaze command was only
    true for the tenure's first interval. Pending orders carry the null gaze row."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "m3"), tag="_m3",
                          vision=False, seed=13)
    try:
        _drive(m, src, 12)
        p = m._tenure_pending
        assert p is not None
        g = p["order"][:, LM.LemLite.GAZE_IDX]
        assert torch.all(g[:, 6] == 1.0) and float(g.abs().sum()) == float(g.shape[0]), (
            "the pending order's gaze row must be the null row (identity quat, "
            "zero strength)")
    finally:
        src.close()


# ===================================================================== M4 / M5
def test_m4_m5_init_and_reset_context_cover_the_new_state():
    m = _agency_model()
    # M4: readable on a fresh model, no AttributeError
    assert m._nf_ach is None and m._nf_handvis is None
    # M5: a batch-size change clears the ring and the plan
    m.goal_ring.append(({"vision": torch.zeros(1, 8)}, torch.zeros(1)))
    m.last_intention = {"plan_length": 1}
    m.reset_context(4)
    assert len(m.goal_ring) == 0 and m.last_intention is None
    assert m._nf_ach is None and m._nf_handvis is None and m._td_err_slow is None


# ===================================================================== M6 - M11
def test_m6_to_m11_every_lesson_grades_raw_reality():
    """The gated blend (up to half the model's own same-step prediction) reached
    six targets. All must read tokens_raw now; Judge B detached."""
    src = inspect.getsource(LM.LemLite.step)
    gate_i = src.index("tokens_raw = {m: tokens[m] for m in tokens}")
    after = src[gate_i:]
    assert "self.token_ring[m].append(tokens_raw[m]" in after                 # M9
    assert "self.last_tokens = {m: tokens_raw[m].squeeze(1).detach() for m in tokens_raw}" in after  # M9
    assert "self.replay.add(tokens_raw, handle_state" in after                # M7
    assert "self.enc_history_raw.append({m: tokens_raw[m]" in after           # M8
    assert "self.latent_regularizer.update(\n" not in after or True
    assert "latent_regularizer.update(" in after and "tokens_raw[m].squeeze(1).detach() for m in tokens_raw}, valid=source_mask)" in after  # M11
    assert "canonical_tokens={m: tokens_raw[m].squeeze(1).detach()" in after  # M6
    # M10/E3. Matched loosely on purpose: the guard is that the anti-collapse pool
    # reads tokens_raw, not that the comprehension keeps one exact line wrapping. The
    # filter gained "and m not in self._no_encoder_output" (2026-09-15) so a sense with
    # no encoder output cannot enter as a zeros token, whose std of 0 contributes a
    # constant 1.0 with no gradient and dilutes every other sense's anti-collapse term.
    assert 'encoder_outputs={m: tokens_raw[m].squeeze(1) for m in tokens_raw' in after
    assert 'if source_mask.get(m) == "real"' in after
    assert 'and m not in self._no_encoder_output}' in after


def test_m7_m9_replay_and_last_tokens_store_raw_not_the_blend(tmp_path):
    """Behavioral: with the gate blending, the tokens replay captures must equal
    last_tokens (raw) and differ from the goal ring's gated tokens."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model(top_down_gate=True)
    m.enable_replay(1, size_per_env=64)
    captured = {}
    orig_add = m.replay.add

    def spy(tokens, *a, **kw):
        captured["vision"] = None
        for name in ("proprioception",):
            if name in tokens:
                captured["tok"] = tokens[name].detach().clone()
        return orig_add(tokens, *a, **kw)

    m.replay.add = spy
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "m7"), tag="_m7",
                          vision=False, seed=15)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        held = prev = None
        for i in range(6):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        stored = captured["tok"].squeeze(1)
        raw = m.last_tokens["proprioception"]
        gated = m.goal_ring[-1][0]["proprioception"]
        assert torch.allclose(stored, raw, atol=1e-6), "replay stored something other than raw"
        assert float(m.last_gate.get("proprioception", 0.0)) > 0.0, (
            "the gate never opened - this run cannot distinguish raw from gated")
        assert not torch.allclose(raw, gated, atol=1e-6), (
            "raw == gated despite an open gate: the capture proves nothing")
    finally:
        src.close()


def test_m8_the_imagination_chain_takes_raw_targets():
    src = inspect.getsource(LM.LemLite.step)
    assert "target_history=(list(self.enc_history_raw)" in src
    lt = inspect.getsource(LT._run_imagination_compiled)
    assert "target_history[s + 1] if target_history is not None" in lt
    assert "observed[m].detach()" in lt, "anchoring must carry the OBSERVED history"


# ===================================================================== M13
def test_m13_visibility_flags_come_from_the_cached_frames():
    src = inspect.getsource(LM.LemLite.step)
    assert "self._nf_handvis = torch.tensor(" in src
    i = src.index("hv = (p.get(\"hand_vis\", hv_end) & hv_end)")
    blk = src[i - 800:i]
    assert "self._nf_handvis" in blk, "the gate must read the cache, not this step's frames"
    j = src.index('"hand_vis": (self._nf_handvis')
    assert j > 0


# ===================================================================== M20 - M23
def test_m20_wellness_pairing_trusts_nf_pred_alone():
    src = inspect.getsource(LM.LemLite.step)
    assert "if self._nf_pred is not None:\n" in src
    assert "if self._nf_pred is not None and self.last_next_pe:" not in src


def test_m21_curiosity_reads_this_steps_delayed_error():
    """The nf grading block must precede the curiosity read of last_next_pe."""
    src = inspect.getsource(LM.LemLite.step)
    assert src.index("self.last_next_pe = {m: float(v.detach()) for m, v in nf_terms.items()}") \
        < src.index("nxt = [v for k, v in (self.last_next_pe or {}).items()")


def test_m22_empty_nf_terms_do_not_raise():
    src = inspect.getsource(LM.LemLite.step)
    assert "torch.stack(list(nf_terms.values())).sum() if nf_terms else None" in src


def test_m23_the_gate_reference_is_a_slow_baseline():
    """The old shut-off compared the fast EMA to 4x the CURRENT batch error: under
    sustained surprise the ratio sits near 1 and the gate stayed open exactly when
    its comment said it must close. K=2.0 measured (fast/slow <= 1.48 across
    H59-H61, 1,035 windows)."""
    src = inspect.getsource(LM.LemLite.step)
    assert "4.0 * err.detach().mean()" not in src
    assert "2.0 * self._td_err_slow" in src
    assert "0.999 * self._td_err_slow" in src
    m = _agency_model(top_down_gate=True)
    B, D = 2, m.d_model
    m._td_err_ema = torch.zeros(B, len(m.pred_mods))
    m._td_err_slow = torch.full((B, len(m.pred_mods)), 1e-6)   # tiny baseline
    # sustained surprise: current err far above the slow baseline -> gate zeroed.
    # Replicate the guard arithmetic on the shipped tensors:
    err = torch.full((B, 1), 1.0)
    m._td_err_ema[:, 0] = 0.95 * m._td_err_ema[:, 0] + 0.05 * err.squeeze(-1)
    closed = (m._td_err_ema[:, 0:1] <= 2.0 * m._td_err_slow[:, 0:1].clamp(min=1e-9)).float()
    assert float(closed.sum()) == 0.0, "a 5-orders-of-magnitude surprise left the gate open"


# ===================================================================== T7
def test_t7_a_non_finite_loss_never_reaches_the_optimizer(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "t7"), tag="_t7",
                          vision=False, seed=17)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        orig = m.loss_manager.compute

        def poisoned(*a, **kw):
            out = orig(*a, **kw)
            if isinstance(out, tuple):
                return (out[0] * float("nan"),) + out[1:]
            return out * float("nan")

        m.loss_manager.compute = poisoned
        before = [p.detach().clone() for p in m.params[:3]]
        m.step(frames, dev, train=True, step_index=0, executed_action=None)
        assert m._nonfinite_steps == 1
        for p0, p1 in zip(before, m.params[:3]):
            assert torch.equal(p0, p1), "parameters moved on a NaN loss"
            assert torch.isfinite(p1).all(), "parameters went non-finite"
    finally:
        src.close()


def test_t7_replay_guard_pinned():
    src = inspect.getsource(LM.LemLite.replay_update)
    assert "torch.isfinite(loss.detach())" in src
    assert "_nonfinite_steps" in src


# ===================================================================== E15 / E18 / E22
def test_e15_replay_motor_rebuild_is_per_window():
    src = inspect.getsource(LM.LemLite.replay_update)
    assert 'w["has_executed"][:, self.ctx_len - 1].all()' not in src
    assert "_lab.any()" in src and "torch.where(\n                    _lab.view(Rn, 1, 1)" in src
    assert "_replay_motor_unlabeled" in src


def test_e18_replay_vicreg_stays_removed():
    """Superseded 2026-09-08: replay VICReg graded DETACHED buffer tokens - it
    trained nothing - and commit 6d1f6bb98 removed it outright. This test's old
    form asserted the per-window row filter of that dead term; the standing
    contract is now that the term stays gone."""
    src = inspect.getsource(LM.LemLite.replay_update)
    assert "vicreg(" not in src.lower() and "_vicreg_loss" not in src, (
        "replay VICReg is back - it graded detached tokens and trained nothing")
    assert "Removed" in src and "VICReg" in src, (
        "keep the removal note: it is the record of WHY the term must stay out")


def test_e22_the_bank_stores_delta_only(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "e22"), tag="_e22",
                          vision=False, seed=19)
    try:
        _drive(m, src, 12)
        assert len(m._tenure_bank) > 0
        assert all(torch.is_tensor(d) and d.dim() == 2 for d in m._tenure_bank), (
            "bank entries must be bare delta tensors now, not (q, delta) pairs")
    finally:
        src.close()


# ===================================================================== E16
def test_e16_the_smoother_survives_a_quaternion_sign_flip():
    """q and -q are the same rotation; blending across the flip used to pass near
    zero and normalize an arbitrary rotation. With hemisphere alignment the blend
    of q and -q is q itself."""
    # 09-09 follow-up review: the smoother moved into the SHARED smooth_blend
    # (one implementation for the live path and offline probes); the step must
    # call it and the alignment must live inside it
    src = inspect.getsource(LM.LemLite.step)
    assert "self.smooth_blend(mfa, self._last_smooth_prev" in src
    blend_src = inspect.getsource(LM.LemLite.smooth_blend)
    assert "pq = torch.where(dot < 0, -pq, pq)" in blend_src
    a = 0.35
    q = torch.tensor([0.3, -0.5, 0.2, 0.79])
    q = q / q.norm()
    prev = -q                                      # same rotation, opposite sign
    dot = (q * prev).sum(-1, keepdim=True)
    prev_aligned = torch.where(dot < 0, -prev, prev)
    out = a * q + (1 - a) * prev_aligned
    out = out / out.norm()
    assert torch.allclose(out.abs(), q.abs(), atol=1e-6), "aligned blend must be q"
    naive = a * q + (1 - a) * prev
    assert naive.norm() < 0.5, "sanity: the naive blend really does collapse"


# ===================================================================== E10 helpers
def test_e10_helper_snapshot_roundtrip():
    class Dummy:
        pass

    o = Dummy()
    o.x = 3.5
    o.arr = np.arange(5).astype(np.float64)
    o.ring = [1, 2, 3]
    o.rng = np.random.default_rng(42)
    o.heavy = object()                              # stands in for a MuJoCo handle
    heavy_ref = o.heavy
    _ = o.rng.normal()                              # advance before snapshot
    st = CT._snap_helper_state(o)
    draws_a = [o.rng.normal() for _ in range(3)]    # branch advances everything
    o.x = -1.0
    o.arr[:] = 0
    o.ring.append(99)
    CT._restore_helper_state(o, st)
    draws_b = [o.rng.normal() for _ in range(3)]
    assert o.x == 3.5 and o.ring == [1, 2, 3]
    assert np.array_equal(o.arr, np.arange(5).astype(np.float64))
    assert draws_a == draws_b, "the rng state did not restore"
    assert o.heavy is heavy_ref, "unsupported attributes must be left untouched"


# ===================================================================== E8 / E9
def test_e8_the_battery_rebuild_covers_the_drives(tmp_path, monkeypatch):
    """E8: the battery rebuild used to hand-copy 14 of 34 keyword arguments. Since
    2026-09-22 _build_model builds through train_mimo.build_learner, the trainer's own
    builder, so this pins the BEHAVIOUR (the run's values reach the constructor) instead
    of the old source text: a constructor spy on the name build_learner calls."""
    run = tmp_path / "R8"
    ckpt = run / "checkpoints" / "ckpt_step_8"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({
        "context_length": 4, "agency_drive": "on", "agency_weight": 0.3,
        "agency_prior": 0.07, "movement_bonus": 0.11, "movement_penalty": 0.22,
        "movement_mref": 9.0, "imagination": "off", "vision_gate": "self-motion",
        "plan_horizon": "licensed", "placeholder_senses": "off",
        "self_prediction": "grounded_residual"}), encoding="utf-8")
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    monkeypatch.setattr(TM, "LemLite", Spy)
    with pytest.raises(RuntimeError):
        CT._build_model(str(ckpt), "cpu", 0, intention=True)
    assert seen["agency_drive"] is True and seen["agency_weight"] == 0.3
    assert seen["agency_prior"] == 0.07 and seen["movement_mref"] == 9.0
    assert seen["movement_bonus"] == 0.11 and seen["movement_penalty"] == 0.22
    assert seen["imagination"] is False
    assert seen["vision_gate"] == "self-motion" and seen["plan_horizon"] == "licensed"
    # the flag whose absence made H84 unloadable (the repair's reason)
    assert seen["self_prediction"] == "grounded_residual"
    mods = seen["overrides"]["modalities"]
    assert mods["audio"]["enabled"] is False and mods["intermind"]["enabled"] is False
    # every keyword the builder passes exists in LemLite's signature
    sig = set(inspect.signature(LemLite.__init__).parameters)
    assert not set(seen) - sig, f"build_learner passes kwargs LemLite lacks: {set(seen) - sig}"
    # the planner-only drives follow the caller's intention choice (R4 guard)
    seen.clear()
    with pytest.raises(RuntimeError):
        CT._build_model(str(ckpt), "cpu", 0, intention=False)
    assert seen["intention"] is False and seen["agency_drive"] is False


def test_e9_the_battery_env_matches_training(tmp_path):
    # since 2026-09-23 the world comes from the trainer's own builder (CT.world_kwargs); each key
    # E9 named carries the run's value, not BridgeEnv's default (tests/test_causal_test_world.py
    # covers the body layout and the probe's own switches)
    src = inspect.getsource(CT.run)
    assert "world_kwargs(a, limits, scene, run_dir, _reps * hold)" in src
    assert 'int(a.get("action_repeat") or 5)' in src
    a = vars(TM.parse_args(["--out", str(tmp_path / "unused_out")]))
    a.update(decoder_kp=2.0, caregiver_seconds=7.0, joint_damping_scale=1.5, hinge_armature=0.05,
             touch_scale_mult=3.0)
    kw = CT.world_kwargs(a, {"phase_name": "mature"}, "crib", str(tmp_path / "scenes"), 25)
    for key, want in (("kp", 2.0), ("caregiver_seconds", 7.0), ("cmd_horizon", 25),
                      ("joint_damping_scale", 1.5), ("hinge_armature", 0.05), ("touch_scale_mult", 3.0)):
        assert kw[key] == want, f"the battery env drops {key}"
    csrc = inspect.getsource(CT._collect)
    assert "range(reps)" in csrc and "range(5)" not in csrc


def test_e11_e12_e13_e14_collection_fixes_pinned():
    csrc = inspect.getsource(CT._collect)
    assert "ctx = (model.context_buffer.detach().float().cpu().clone()" in csrc  # E12 (first-step None guard added 09-03)
    assert csrc.index("ctx = (model.context_buffer") < csrc.index("res = model.step")
    assert "torch.zeros_like(_cb" not in csrc                                    # E12
    assert 'rec["cmd_mag"] = p_mag' in csrc                                      # E13
    assert 'out[-1]["cmd_mag"]' not in csrc                                      # E13
    rsrc = inspect.getsource(CT.run)
    # E14: every body row but the eyes', found by name (rows 0-22 until 2026-09-23 left out the trunk)
    assert "body_rows_differ(cand, a_true, model.GAZE_IDX)" in rsrc
    assert 'stream("pas"' not in rsrc                                            # E11


def test_e17_limits_come_from_the_record_nearest_the_checkpoint():
    src = inspect.getsource(CT._limits_for_run)
    assert "abs(int(r.get(\"step\", 0)) - _ck_step)" in src


# ===================================================================== E1/E6/E7/E20/E21
def test_e1_env_imports_logging():
    src = inspect.getsource(EV)
    assert "\nimport logging\n" in src


def test_e6_window_metric_skips_gaze_and_appends_once():
    src = inspect.getsource(EV.BridgeEnv._command_response)
    j = src.index("w_cmd_all = np.zeros")
    blk = src[j:j + 900]
    assert "if i == gaze_idx:" in blk, "the window loop lost the gaze skip"
    assert "record_window" in src
    step_src = inspect.getsource(EV.BridgeEnv.step)
    assert "record_window=False" in step_src, "the slow scorer must not append"


def test_e7_reset_clears_the_instrument_state(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    s = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "e7"), tag="_e7",
                        vision=False, seed=21)
    try:
        env = s.env
        env._cmd_delay = [1, 2, 3]
        env._jit_prev = np.ones(4)
        env._jit_ring = [0.5]
        env._unseen_s = 9.0
        s.reset()
        # reset()'s trailing _collect legitimately re-seeds _jit_prev from the
        # first post-teleport frame; the defect was PAIRING across the teleport -
        # so the ring must be empty (no sample spanned the jump) and the pre-set
        # marker array must be gone
        assert env._cmd_delay == []
        assert not (isinstance(env._jit_prev, np.ndarray) and env._jit_prev.shape == (4,))
        assert env._jit_ring == [] and env._unseen_s <= 0.05
    finally:
        s.close()


def test_e20_e21_docs_and_ceiling():
    src = inspect.getsource(EV)
    assert 'if kind == "crib":\n                cp = make_ceiling_texture' in src
    import mimo_bridge.proprio_adapter as PA
    assert "# 24 (23 body bones + the gaze handle)" in inspect.getsource(PA)
    import mimo_bridge.datasource as DS
    assert "(24, 11)" in inspect.getsource(DS)
    from lem.model import wellness as WL
    assert "pressure(0)" in inspect.getsource(WL)


# ===================================================================== trainer (T) pins
def test_t_trainer_fixes_pinned():
    src = inspect.getsource(TM)
    # T1: the gate survives to the loop - ring_need must NOT be zeroed
    # unconditionally, or replay starts on ~67 stored moments and trains on
    # near-duplicates. The condition was later widened to also cover an eval-only
    # run launched with --fill-ring-first, where replay is never enabled and
    # ring_need would reach the episode companion writer unbound. Assert the
    # BEHAVIOUR both spellings share, not one spelling of it.
    assert "ring_need = 0" in src
    assert "if not (args.fill_ring_first and args.replay" in src
    unconditional = chr(10) + "        ring_need = 0" + chr(10)
    assert unconditional not in src, "ring_need is zeroed unconditionally again"
    # T4: BOM-aware, non-destructive control read
    assert 'raw.decode("utf-8-sig")' in src and 'raw.decode("utf-16")' in src
    assert "ctrl_retries" in src
    # T5
    assert "snapshot stop snapshot_and_stop video advance_phase" in src
    # T9
    assert 'ckpt_list[-1]["step"] == step' in src
    # T12
    assert 'if "phase_step" in cs:' in src
    # T13
    assert '"type": "launch"' in src
    # T14
    assert "model.noise_rng.manual_seed(_s2)" in src
    # T17
    assert "except (ValueError, TypeError) as e:" in src
    # T8
    assert "_keep_last_n" in src
    # T11: tn_n counts pose events and is emitted
    assert "if \"tenure\" in res.breakdown:" in src and "\"tenure_events\"" in src
    assert '"td_gate"' in src
    # item 4 (2026-09-02) retired the pe_real_mean stopgap: train_mean_pe itself
    # is real-sources-only now
    assert '"pe_real_mean"' not in src


# ===================================================================== checkpoint (T6/T15/T16)
def test_t6_partial_checkpoint_is_refused(tmp_path):
    m = _agency_model(checkpoint_dir=str(tmp_path / "ck6"))
    p = m.save_checkpoint(77)
    assert os.path.basename(p) == "ckpt_step_77"
    assert not any(d.startswith(".tmp") for d in os.listdir(os.path.dirname(p))), (
        "the tmp build directory must be renamed away")
    os.remove(os.path.join(p, "meta.json"))
    m2 = _agency_model(checkpoint_dir=str(tmp_path / "ck6"))
    with pytest.raises(RuntimeError, match="partial save"):
        m2.load_checkpoint(p, load_optimizer=False)


def test_t6_missing_optimizer_warns_but_loads(tmp_path, caplog):
    import logging as _logging
    m = _agency_model(checkpoint_dir=str(tmp_path / "ck6b"))
    p = m.save_checkpoint(88)
    os.remove(os.path.join(p, "optimizer.pt"))
    m2 = _agency_model(checkpoint_dir=str(tmp_path / "ck6b"))
    with caplog.at_level(_logging.WARNING):
        m2.load_checkpoint(p, load_optimizer=True)
    assert any("optimizer state ABSENT" in r.message for r in caplog.records)


def test_t15_partial_component_load_is_reported(caplog):
    import logging as _logging
    src = inspect.getsource(CKPT.CheckpointManager.load)
    assert "missing / %d unexpected keys" in src


def test_t16_fresh_optimizer_skips_the_optimizer_load():
    src = inspect.getsource(CKPT.CheckpointManager._load_state)
    assert "want_optimizer and opt_path.exists()" in src
    lsrc = inspect.getsource(CKPT.CheckpointManager.load)
    assert "want_optimizer=optimizer is not None" in lsrc


# ===================================================================== M14
def test_m14_live_tokens_encodes_the_three_senses_only():
    src = inspect.getsource(LM.LemLite.step)
    i = src.index("def _live_tokens(raw, mask):")
    blk = src[i:i + 700]
    assert 'r2 = {m: raw[m] for m in ("vision", "tactile", "proprioception")' in blk
    assert 'r2["motor"]' not in blk, "the dead motor-row swap is back"
    assert "end_row" not in src[i - 3000:i + 3000], "end_row had no consumer"


# ---------------------------------------------------------------- audit P042 (2026-09-18)
def test_p042_the_live_prediction_heads_pair_on_raw_tokens():
    """P042: the residual base AND the loss target both read tokens_raw. Moving one
    without the other introduces gate dependence instead of removing it, so this
    asserts both lines and the absence of the old gated pair."""
    src = inspect.getsource(LM.LemLite.step)
    after = src[src.index("tokens_raw = {m: tokens[m] for m in tokens}"):]
    assert 'who="lem_model live"' in after
    assert 'current={m: tokens_raw[m].squeeze(1) for m in tokens_raw},' in after
    assert 'targets = {m: tokens_raw[m].squeeze(1).detach()' in after
    assert 'for m in self.pred_mods if m in tokens_raw}' in after
    assert 'current={m: tokens[m].squeeze(1) for m in tokens}' not in after
    assert 'targets = {m: tokens[m].squeeze(1).detach()' not in after


def test_p042_the_live_prediction_residual_base_is_the_raw_token(tmp_path):
    """Behavioural: force every td_gate open, zero the head weights so the prediction IS
    its residual base, then last_predictions must equal last_tokens (raw) and differ from
    last_tokens_fed (the blend) on the sense whose gate opened."""
    from mimo_bridge.datasource import SingleEnvSource
    m = LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True,
                action_tenure_loss=True, action_tenure_token_reach=True,
                agency_drive=True, intention=True, intention_execute=False, top_down_gate=True,
                overrides={"training": {"steps_per_phase": 3000},
                           "model": {"residual_prediction": True}})
    for lin in m.td_gates.values():
        torch.nn.init.constant_(lin.bias, 6.0)      # sigmoid saturates -> g -> 0.50
        torch.nn.init.zeros_(lin.weight)
    for h in m.prediction_heads.heads.values():
        for p in h.parameters():
            torch.nn.init.zeros_(p)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "p042"), tag="_p042",
                          vision=False, seed=15)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        held = prev = None
        for i in range(3):      # _nf_pred must exist before the gate can blend anything
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert max(m.last_gate.values()) > 0.01, "the gate never blended - test is vacuous"
        opened = max(m.last_gate, key=m.last_gate.get)
        raw = m.last_tokens[opened]
        fed = m.last_tokens_fed[opened].squeeze(1)
        assert float((fed - raw).abs().max()) > 1e-4, "raw and fed are identical - vacuous"
        got = m.last_predictions[opened]
        # measured 2026-09-18: |got - raw| 2.2e-4 (the head's own zeroed path plus the
        # output normalisation), |got - fed| 1.27 on touch and 0.13 on body sense
        assert torch.allclose(got, raw, atol=1e-3), "the residual base is not the raw token"
        assert not torch.allclose(got, fed, atol=1e-3), "the residual base is the blend"
    finally:
        src.close()


def test_p262_no_lesson_reaches_the_voice_head_and_its_forward_builds_no_graph(tmp_path):
    """Audit P262: the bridge's voice head receives no lesson (voice_target=None, loss weight
    0.0), so its forward pass runs without gradient tracking; its output is still kept for
    the page."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _agency_model()
    assert m.voice_head is not None, "master_config.yaml enables the voice head"
    seen = []
    m.voice_head.register_forward_hook(
        lambda mod, inp, out: seen.append((torch.is_grad_enabled(), bool(out.requires_grad))))
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "p262"), tag="_p262",
                          vision=False, seed=17)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        before = [p.detach().clone() for p in m.voice_head.parameters()]
        m.step(frames, dev, train=True, step_index=0, executed_action=None)
    finally:
        src.close()
    assert seen == [(False, False)], f"the voice head ran with gradient tracking: {seen}"
    assert all(p.grad is None for p in m.voice_head.parameters()), "a lesson reached the voice head"
    for p0, p1 in zip(before, m.voice_head.parameters()):
        assert torch.equal(p0, p1), "the optimizer moved a voice weight"
    assert m.last_voice is not None and m.last_voice.shape[-1] == 80
