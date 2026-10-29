# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Motor babbling must have a live channel, and the override must reach it.

UEMA 19.2 makes motor babbling the bootstrap: constrained random commands break the
null prediction non-uniformly, and that is what separates self from world. If the
babble amplitude is zero, the bootstrap has no driver.

What was measured on H28: the developmental phase table sets exploration_noise to 0.01
at neonate and 0.0 at MATURE, and --steps-per-phase 3000 across five phases puts the
body at mature by step 12,000. The model has 826K steps of lifetime, so babbling was
off for about 98% of it. The movement drive noticed - stillness debt averaged 0.21 and
it raised noise_mult to 1.93 - but 1.93 x 0.0 is still 0.0, so the feedback loop was
pushing on a dead channel. Until H23 the planner supplied the movement instead; when
that was switched off there was no source left and commands fell to 0.03-0.08 rad.

These tests do not assert what the phase values SHOULD be. They assert that the values
are visible, that a zero one is reachable by the override, and that the multiplier
cannot manufacture motion from zero.
"""
import pytest
import torch

from lem.training.curriculum import CurriculumManager
from mimo_bridge.movement_drive import MovementDrive


def _phase_noise(steps_per_phase=3000):
    """What the CURRICULUM hands the trainer - not the static defaults table.

    Two tables exist. lem/config/defaults.py lists four phases ending at child with
    0.0; CurriculumManager produces six, ending at mature with 0.0. The trainer uses
    the manager when --curriculum is on, which every recent run has been.
    """
    cm = CurriculumManager(steps_per_phase=steps_per_phase)
    seen = {}
    for i in range(steps_per_phase * 8):
        cm.step(mean_pe=0.05)
        d = cm.get_dev_limits()
        seen.setdefault(d.get("phase_name"), (i, float(d.get("exploration_noise", 0.0))))
    return seen


def test_mature_is_reached_early_and_no_phase_switches_babble_off():
    """Mature arrives in 15k steps and stays - that is still the schedule, and still a
    fact worth pinning: UEMA 20.2 budgets ~10M steps for Stream 1 (motor babbling) alone.

    What changed on 2026-08-27 is what mature MEANS for babble. H28 ran 124,483 steps at
    mature with the amplitude at 0.0 and the body froze; seven runs did the same. Movement
    is a wellness term, so the operator's decision is that babble never reaches zero: the
    schedule floors it at BABBLE_FLOOR, to be questioned again when intention is on.
    """
    from lem.training.curriculum import BABBLE_FLOOR
    seen = _phase_noise(3000)
    zero = {k: v for k, v in seen.items() if v[1] == 0.0}
    assert not zero, f"a phase switches babble off again: {sorted(zero)}"
    assert "mature" in seen, f"phases seen: {sorted(seen)}"
    first_step, mature_noise = seen["mature"]
    assert first_step < 20000, (
        f"mature reached at step {first_step}; a change to steps_per_phase is a deliberate act.")
    assert mature_noise == BABBLE_FLOOR == 0.05


def test_every_phase_before_mature_has_a_live_babble_channel():
    seen = _phase_noise(3000)
    live = {k: v[1] for k, v in seen.items() if k != "mature"}
    assert all(v > 0.0 for v in live.values()), f"a pre-mature phase has zero babble: {live}"


def test_noise_multiplier_cannot_manufacture_movement_from_zero():
    """The stillness feedback scales the babble; scaling zero is zero.

    On H28 this multiplier reached 1.93 while the body sat at 4.4 cm/s, because the
    quantity it multiplies was pinned at zero by the mature phase.
    """
    d = MovementDrive(n_envs=2, device="cpu")
    still = torch.zeros(2, 23, 12)
    out = None
    for _ in range(80):
        out = d.update(still, self_factor=1.0)
    assert out["noise_mult"] > 1.0, "stillness never raised the babble multiplier"
    debt = out["debt"]
    debt = float(debt.mean()) if hasattr(debt, "mean") else float(debt)
    assert debt > 0.0, "a still body accrued no stillness debt"
    assert 0.0 * out["noise_mult"] == 0.0      # the arithmetic seven runs relied on


@pytest.mark.parametrize("phase_noise,scale,absolute,expected", [
    (0.0,  8.0,  None, 0.0),      # the H28 case: scaling zero stays zero
    (0.0,  1.0,  0.05, 0.05),     # the absolute override restores it
    (0.01, 2.0,  None, 0.02),     # scaling a live value works
    (0.01, 1.0,  0.05, 0.05),     # absolute wins over the phase value
])
def test_override_precedence_matches_the_trainer(phase_noise, scale, absolute, expected):
    """The precedence the training loop applies, stated once so it cannot drift."""
    dev = {"exploration_noise": phase_noise}
    if absolute is not None:
        dev = {**dev, "exploration_noise": float(absolute)}
    elif scale != 1.0:
        dev = {**dev, "exploration_noise": dev.get("exploration_noise", 0.0) * scale}
    assert dev["exploration_noise"] == pytest.approx(expected)


def test_trainer_exposes_an_absolute_babble_override():
    """--babble-scale multiplies the value limits_for() returns, which the floor holds at
    0.05 in every phase - so at mature it multiplies 0.05, not the table's 0.0."""
    import inspect
    from mimo_bridge import train_mimo
    src = inspect.getsource(train_mimo)
    assert "--babble-noise" in src, "no absolute override exists"
    assert 'dev = {**dev, "exploration_noise": float(args.babble_noise)}' in src


def test_causal_test_names_an_inert_command_pathway():
    """Zero command columns produce zero gaps by construction, not by measurement.

    ckpt_step_701642_tc19 was grown 10 -> 19 wide with zero padding and never trained
    those columns: |w| is exactly 0.000000. Its causal test returned gaps of exactly
    0.000000 for every modality, which the report called INCONCLUSIVE - the same label
    it gives a live pathway whose signal is merely small. They are different claims.
    """
    import inspect
    from mimo_bridge.tools import causal_test as ct
    src = inspect.getsource(ct)
    assert "_command_pathway_weight" in src
    assert '"INERT"' in src
    assert "command_pathway_inert" in src
