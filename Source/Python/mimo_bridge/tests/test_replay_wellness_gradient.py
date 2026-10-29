# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Re-study hands the motor controller the wellness slope, rebuilt from the ring (P166).

Before 2026-09-18 replay_update called the controller with no wellness_grad, so
MotorController.forward substituted zeros into 234 of its 1,262 inputs (18.54%) and
re-study trained those columns at exactly zero gradient while the live step acted and
graded with real values (H83: mean |slope| 0.000426 to 0.004861, above zero in all 137
windows). Nothing new is stored: the slope is a parameter-free function of the achieved
pose and the window's rom limit, both already in the ring. Behavioural tests - they
read values, never inspect.getsource.
"""
import torch

from mimo_bridge.lem_model import LemLite


def _model():
    return LemLite(device="cpu", seed=0, imagination=False, think_harder=True,
                   motor_state_formula="boundary")


def _fill_ring_with_a_boundary_pose(model, n=40):
    """Positions centred on the handle centroid; one handle sits inside the last 10% of
    the reach envelope (0.189..0.21 per coordinate at rom 1.0), the rest inside it."""
    torch.manual_seed(3)
    for _ in range(n):
        toks = {m: torch.randn(2, 1, model.d_model) * 0.1 for m in model.active}
        ach = torch.zeros(2, len(model.handle_names), 10)
        ach[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ach[:, 0, 0] = 0.205          # centroid-relative 0.1965 -> in the band
        ach[..., 8] = 0.5
        model.replay.add(toks, ach, {"rom_limit": 1.0, "strength_limit": 1.0})


def test_replay_hands_the_controller_the_slope_instead_of_zeros():
    model = _model()
    model.enable_replay(n_envs=2, size_per_env=64)
    _fill_ring_with_a_boundary_pose(model)
    seen = {}
    orig = model.motor_controller.forward

    def spy(latents, handle_state, wellness_grad=None, **kw):
        seen["wg"] = None if wellness_grad is None else wellness_grad.detach().clone()
        return orig(latents, handle_state, wellness_grad=wellness_grad, **kw)

    model.motor_controller.forward = spy
    out = model.replay_update(batch=6)
    assert out is not None
    wg = seen.get("wg")
    assert wg is not None, "replay still calls the controller without a wellness slope"
    assert wg.shape[1:] == (len(model.handle_names), 9)
    assert float(wg.abs().max()) > 0.0, "the rebuilt slope is all zeros on a boundary pose"


def test_the_replay_slope_matches_the_live_definition():
    model = _model()
    B, H = 3, len(model.handle_names)
    torch.manual_seed(5)
    pose = torch.randn(B, H, 9) * 0.05
    pose[:, 1, 1] = 0.2                                   # a coordinate inside the band
    pose[..., 0:3] = pose[..., 0:3] - pose[..., 0:3].mean(dim=1, keepdim=True)
    # live construction, as lem_model.step builds it
    live_cmd = pose.clone().requires_grad_(True)
    score = model.wellness._formula_motor_state(live_cmd, B, live_cmd.device)
    live = torch.autograd.grad(score.sum(), live_cmd, create_graph=True, allow_unused=True)[0]
    # the replay construction, per-sample limit equal to the module's scalar
    rep_cmd = pose.clone().requires_grad_(True)
    rebuilt = model.wellness.motor_command_gradient(rep_cmd, rom_limit=torch.full((B,), model.wellness._rom_limit))
    assert float((live.detach() - rebuilt).abs().max()) == 0.0
    assert float(rebuilt.abs().max()) > 0.0


def test_a_pose_clear_of_the_limits_still_gives_zeros():
    model = _model()
    B, H = 2, len(model.handle_names)
    pose = torch.zeros(B, H, 9)
    pose[..., 0:3] = torch.randn(B, H, 3) * 0.02          # far inside the envelope
    pose[..., 0:3] = pose[..., 0:3] - pose[..., 0:3].mean(dim=1, keepdim=True)
    cmd = pose.requires_grad_(True)
    g = model.wellness.motor_command_gradient(cmd, rom_limit=torch.full((B,), 1.0))
    assert float(g.abs().max()) == 0.0


def test_the_scalar_limit_path_is_unchanged():
    """Regression guard for the branch's straight-through penalty: the rom_limit=None
    branch and a per-sample tensor equal to the scalar give the same forward value,
    the forward is bit-identical to the clamped form, and the slope past the limit is
    the band-end value held (-6.66667 at rom 1.0)."""
    model = _model()
    w = model.wellness
    B, H = 1, len(model.handle_names)
    pos = torch.linspace(0.0, 3.0 * 0.21, 601)
    vals, slopes = [], []
    for p in pos:
        cmd = torch.zeros(B, H, 9)
        cmd[0, 0, 0] = float(p)
        cmd.requires_grad_(True)
        s_none = w._formula_motor_state_boundary(cmd)
        s_tensor = w._formula_motor_state_boundary(cmd, rom_limit=torch.full((B,), w._rom_limit))
        assert float((s_none - s_tensor).abs().max()) == 0.0
        # clamped reference: penalty = min(q,1)^2 on the same coordinate
        rom = w._rom_limit * 0.21
        q = max(0.0, 0.1 * rom - (rom - float(p))) / (0.1 * rom)
        ref = 1.0 - (min(q, 1.0) ** 2) / (H * 3)
        assert abs(float(s_none) - ref) < 1e-6
        (g,) = torch.autograd.grad(s_none.sum(), cmd)
        vals.append(float(s_none)); slopes.append(float(g[0, 0, 0]))
    # past the limit the slope is HELD at the band-end value: d(q^2)/d|pos| at q = 1 is
    # 2/threshold per coordinate, divided by the H x 3 coordinates the mean runs over
    rom = w._rom_limit * 0.21
    held = -2.0 / (0.1 * rom) / (H * 3)
    past = [s for p, s in zip(pos, slopes) if float(p) > rom * 1.05]
    assert past, "no sample past the limit"
    assert all(abs(s - held) < 1e-4 for s in past), (past[:3], held)
    inside = [s for p, s in zip(pos, slopes) if float(p) < rom * 0.85]
    assert all(s == 0.0 for s in inside), "the interior must keep a zero slope"


def test_the_slope_columns_receive_gradient_in_re_study():
    """The 234 wellness-slope input columns of the controller's first layer read exactly
    0.000e+00 gradient from a replay pass before the fix; with a boundary pose in the
    ring they must now be nonzero."""
    model = _model()
    model.enable_replay(n_envs=2, size_per_env=64)
    _fill_ring_with_a_boundary_pose(model)
    W = model.motor_controller.input_proj.weight
    d = model.motor_controller._wellness_grad_dim
    grads = {}
    orig_step = model.optimizer.step

    def spy_step(*a, **k):
        grads["cols"] = W.grad[:, -d:].detach().abs().sum().item()
        grads["rest"] = W.grad[:, :-d].detach().abs().sum().item()
        return orig_step(*a, **k)

    model.optimizer.step = spy_step
    out = model.replay_update(batch=6)
    assert out is not None and "cols" in grads
    assert grads["rest"] > 0.0
    assert grads["cols"] > 0.0, "the slope columns still receive no gradient from re-study"
