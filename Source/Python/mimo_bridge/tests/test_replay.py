# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The diary (replay memory): window sampling is exact, and one re-study step learns.

Hand-worked checks:
  * tokens are stamped with their step number, so a sampled window must read as
    consecutive integers - any ring-wrap or episode-crossing bug breaks that;
  * a window must never mix two episodes (mark_reset splits them);
  * LemLite.replay_update on CPU returns a finite loss and actually moves weights.
"""
import torch

from mimo_bridge.replay import ReplayMemory

MODS = ["a", "b"]
D = 8


def _fill(mem, n_steps, n_envs, start=0):
    for t in range(start, start + n_steps):
        toks = {m: torch.full((n_envs, 1, D), float(t)) for m in MODS}
        ach = torch.full((n_envs, 24, 10), float(t))
        mem.add(toks, ach, {"rom_limit": 0.5, "strength_limit": 0.25})


def test_windows_are_consecutive_and_single_episode():
    mem = ReplayMemory(MODS, D, size_per_env=64, n_envs=2)
    _fill(mem, 10, 2)          # episode 0: steps 0..9
    mem.mark_reset()
    _fill(mem, 10, 2, 10)      # episode 1: steps 10..19
    w = mem.sample_windows(batch=16, ctx_len=4, device="cpu")
    assert w is not None
    stamps = w["tokens"][:, :, 0, 0]              # [R, need] step stamps
    for row in stamps:
        diffs = row[1:] - row[:-1]
        assert torch.all(diffs == 1.0), f"window not consecutive: {row.tolist()}"
        assert not (row.min() < 10 <= row.max()), f"window crosses the reset: {row.tolist()}"
    assert float(w["rom"][0]) == 0.5 and float(w["strength"][0]) == 0.25


def test_ring_wrap_still_consecutive():
    mem = ReplayMemory(MODS, D, size_per_env=12, n_envs=1)
    _fill(mem, 30, 1)          # wraps the ring 2.5 times; live steps are 18..29
    w = mem.sample_windows(batch=12, ctx_len=4, device="cpu")
    assert w is not None
    for row in w["tokens"][:, :, 0, 0]:
        assert torch.all(row[1:] - row[:-1] == 1.0), f"wrap broke order: {row.tolist()}"
        assert row.min() >= 18, "sampled a step the ring already forgot"


def test_too_little_history_returns_none():
    mem = ReplayMemory(MODS, D, size_per_env=64, n_envs=1)
    _fill(mem, 3, 1)
    assert mem.sample_windows(batch=8, ctx_len=4, device="cpu") is None


def test_past_motor_reencoding_does_not_retain_an_unused_backward_graph():
    from mimo_bridge.lem_model import LemLite
    model=LemLite(device='cpu',seed=0,imagination=False)
    model.enable_replay(2,size_per_env=32)
    for i in range(12):
        tokens={m:torch.randn(2,1,model.d_model) for m in model.active}
        command=torch.zeros(2,24,9);command[...,6]=1
        model.replay.add(tokens,torch.zeros(2,24,10),{},executed=command)
    calls=[];original=model.encoder_registry.encode
    def observe(frame,mask):
        if set(frame)=={'motor'}: calls.append((frame['motor'].shape[0],torch.is_grad_enabled()))
        return original(frame,mask)
    model.encoder_registry.encode=observe
    result=model.replay_update(batch=6)
    assert result is not None
    assert (6,True) in calls  # the current outgoing command still learns
    assert (6*(model.ctx_len-1),False) in calls  # historical context has no gradient contract


def test_replay_update_learns_on_cpu():
    from mimo_bridge.lem_model import LemLite
    model = LemLite(device="cpu", seed=0)
    model.enable_replay(n_envs=2, size_per_env=64)
    mem = model.replay
    torch.manual_seed(1)
    for t in range(12):
        toks = {m: torch.randn(2, 1, model.d_model) * 0.1 for m in model.active}
        ach = torch.randn(2, 24, 10) * 0.1
        mem.add(toks, ach, {"rom_limit": 1.0, "strength_limit": 1.0})
    before = model.motor_controller.mlp[0].weight.detach().clone() if hasattr(model.motor_controller, "mlp") \
        else next(model.motor_controller.parameters()).detach().clone()
    out = model.replay_update(batch=6)
    assert out is not None and torch.isfinite(torch.tensor(out["replay_loss"])), out
    after = model.motor_controller.mlp[0].weight.detach() if hasattr(model.motor_controller, "mlp") \
        else next(model.motor_controller.parameters()).detach()
    assert not torch.equal(before, after), "optimizer step did not move the motor controller"


def test_replay_trains_the_motor_encoder_and_no_other_encoder():
    """Re-study reaches exactly one encoder. The explainer page said it reaches none.

    Five senses' tokens come out of the ring detached, so their encoders receive no
    gradient. The motor token is NOT replayed from storage: replay_update rebuilds it
    from [achieved pose, outgoing executed order] through the live encoder registry
    with gradients on, so the motor encoder is trained from memory as well.

    Behavioural, not a source-text match: it runs the pass and reads .grad. The six
    guards that assert inspect.getsource() substrings pass on a behaviour change that
    keeps the string and fail on a whitespace-only reformat that keeps the behaviour.
    """
    from mimo_bridge.lem_model import LemLite
    model = LemLite(device="cpu", seed=0, imagination=False)
    nh = len(model.handle_names)
    model.enable_replay(n_envs=2, size_per_env=64)
    torch.manual_seed(1)
    for _ in range(40):
        toks = {m: torch.randn(2, 1, model.d_model) * 0.5 for m in model.active}
        ex = torch.randn(2, nh, 9) * 0.5
        ex[..., 6] = 1.0
        model.replay.add(toks, torch.randn(2, nh, 10) * 0.1,
                         {"rom_limit": 1.0, "strength_limit": 1.0}, executed=ex)
    for p in model.params:
        p.grad = None
    assert model.replay_update(batch=6) is not None
    grads = {m: sum(0.0 if p.grad is None else float(p.grad.abs().sum())
                    for p in enc.parameters())
             for m, enc in model.encoder_registry.encoders.items()}
    assert grads["motor"] > 0.0, (
        "the motor encoder got no gradient from re-study - its token is supposed to be "
        "rebuilt from [achieved pose, executed order] with gradients on")
    for m, g in grads.items():
        if m == "motor":
            continue
        assert g == 0.0, (
            m + "'s encoder was trained by re-study; its tokens were stored detached")
    # P411: the publishable invariant is the COUNT and the weight move, never the sum
    enc_motor = model.encoder_registry.encoders["motor"]
    for p in enc_motor.parameters():
        assert p.grad is not None and int((p.grad != 0).sum()) == p.numel(), \
            "every one of the motor encoder's numbers takes gradient from re-study"
    lr = model.optimizer.param_groups[0]["lr"]
    before = {n: q.detach().clone() for n, q in enc_motor.named_parameters()}
    model.optimizer.step()   # the pass above already ran one step; a second measures the move size
    moved = max(float((q.detach() - before[n]).abs().max()) for n, q in enc_motor.named_parameters())
    assert abs(moved / lr - 1.0) < 0.05, (moved, lr)


def test_the_motor_encoders_gradient_sum_is_a_ring_draw_not_a_property_of_the_code():
    """P411: a sum measured from a random ring at one batch may not be published as a
    property of the code. Same ring seed, two batch sizes: identical nonzero COUNTS,
    different sums."""
    from mimo_bridge.lem_model import LemLite

    def one(batch):
        model = LemLite(device="cpu", seed=0, imagination=False)
        nh = len(model.handle_names)
        model.enable_replay(n_envs=2, size_per_env=64)
        torch.manual_seed(1)
        for _ in range(40):
            toks = {m: torch.randn(2, 1, model.d_model) * 0.5 for m in model.active}
            ex = torch.randn(2, nh, 9) * 0.5
            ex[..., 6] = 1.0
            model.replay.add(toks, torch.randn(2, nh, 10) * 0.1,
                             {"rom_limit": 1.0, "strength_limit": 1.0}, executed=ex)
        for p in model.params:
            p.grad = None
        assert model.replay_update(batch=batch) is not None
        enc = model.encoder_registry.encoders["motor"]
        count = sum(int((p.grad != 0).sum()) for p in enc.parameters())
        total = sum(float(p.grad.abs().sum()) for p in enc.parameters())
        return count, total

    c6, s6 = one(6)
    c12, s12 = one(12)
    assert c6 == c12 > 0
    assert abs(s6 - s12) / max(s6, s12) > 1e-6


def test_the_write_head_is_already_outside_every_window_the_sampler_can_draw():
    """The branch that rejects a window spanning the write head cannot fire.

    sample_windows draws its start with randint(count - need), an offset from the OLDEST
    slot, so the furthest a window reaches is offset size-2 while the write head sits at
    offset size-1. This sweeps the real draw over every value it can return and reads the
    step stamps back out: the newest stored moment appears in no window at all, so the
    guard has nothing left to reject. The property pinned here is the START RANGE, which
    is what actually protects the sampler. Widen it by one - randint(count - need + 1) -
    and the newest moment starts appearing, this test fails, and whoever widened it has
    to make the guard carry the protection instead. The guard's predicate must stay
    idx[:-1], the window INTERIOR: a window that ENDS on the newest slot is still a run
    of consecutive moments.
    """
    size, ctx = 64, 4
    mem = ReplayMemory(MODS, D, size_per_env=size, n_envs=1)
    _fill(mem, size + 17, 1)                   # wrap, so count == size
    assert mem.count == size
    need = ctx + 1
    newest_stamp = float(size + 17 - 1)
    oldest_stamp = newest_stamp - (size - 1)

    real_randint = torch.randint
    sweep = {"next": 0, "draws": [], "n": None}

    def stub(n, shape, *a, **kw):
        if int(n) == mem.n_envs:               # the body draw
            return real_randint(n, shape, *a, **kw)
        sweep["n"] = int(n)
        v = sweep["next"] % int(n)
        sweep["next"] += 1
        sweep["draws"].append(v)
        return torch.tensor([v])

    import mimo_bridge.replay as replay_mod
    replay_mod.torch.randint = stub
    try:
        w = mem.sample_windows(batch=size + 8, ctx_len=ctx, device="cpu")
    finally:
        replay_mod.torch.randint = real_randint

    assert w is not None
    assert sorted(set(sweep["draws"])) == list(range(sweep["n"])), (
        "the sweep did not cover every offset the draw can return")
    stamps = w["tokens"][:, :, 0, 0]
    for row in stamps:
        assert torch.all(row[1:] - row[:-1] == 1.0), f"window not consecutive: {row.tolist()}"
    assert float(stamps.max()) == newest_stamp - 1.0, (
        f"a window reached step {float(stamps.max())} while the newest stored moment is "
        f"{newest_stamp}: the start range no longer stops short of the write head, so "
        f"the guard has to carry that protection now")
    assert float(stamps.min()) == oldest_stamp


def test_the_window_shapes_follow_the_body_rather_than_a_24_handle_default():
    """The module docstring sends a reader here for the ring's contract. trunk_v1 has 26
    handles, so a comment reading 24 tells that reader the wrong shape."""
    import inspect
    import mimo_bridge.replay as replay_mod

    nh = 26
    mem = ReplayMemory(MODS, D, n_handles=nh, size_per_env=32, n_envs=1)
    for t in range(12):
        mem.add({m: torch.full((1, 1, D), float(t)) for m in MODS},
                torch.full((1, nh, 10), float(t)),
                {"rom_limit": 1.0, "strength_limit": 1.0},
                executed=torch.full((1, nh, 9), float(t)))
    assert mem.achieved.shape[-2] == nh and mem.executed.shape[-2] == nh
    w = mem.sample_windows(batch=8, ctx_len=4, device="cpu")
    assert w is not None
    assert w["achieved"].shape[-2] == nh and w["executed"].shape[-2] == nh

    # no shape comment in the file may name a fixed handle count
    src = inspect.getsource(replay_mod)
    for marker in ("24, 10]", "24, 9]", "[B, 24", "[need, 24", "ctx+1, 24"):
        assert marker not in src, f"a stale {marker} shape is still documented here"
