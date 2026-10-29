# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Helpers of the latent body causal-control test (mimo_bridge.tools.latent_body_intervention)."""
import json

import torch

from lem.latent_body.learning import ShadowLearner
from mimo_bridge.tests.latent_body_fixtures import config, tree
from mimo_bridge.tools import latent_body_intervention as LI


def _inputs(learner):
    state = {n.name: torch.randn(1, len(learner.tree.subtree[n.name]) * 12) for n in learner.tree.nodes}
    recent = {n.name: torch.randn(1, len(learner.tree.subtree[n.name]) * 12) for n in learner.tree.nodes}
    return state, recent


def test_decode_with_no_override_is_the_hierarchys_own_decode():
    learner = ShadowLearner(tree(), config())
    state, recent = _inputs(learner)
    roots = {r: torch.randn(1, learner.model.latent_dim) for r in learner.tree.roots}
    with torch.no_grad():
        own, _ = learner.model.decode(state, recent, roots)
        ours = LI.decode_with(learner.model, state, recent, roots, {})
    for name in own:
        assert torch.equal(own[name].change, ours[name].change)


def test_an_override_changes_only_that_part_and_its_descendants():
    learner = ShadowLearner(tree(), config())
    state, recent = _inputs(learner)
    roots = {r: torch.zeros(1, learner.model.latent_dim) for r in learner.tree.roots}
    child = next(n for n in learner.tree.nodes if n.parent is not None)
    with torch.no_grad():
        base = LI.decode_with(learner.model, state, recent, roots, {})
        moved = LI.decode_with(learner.model, state, recent, roots,
                               {child.name: torch.full((1, learner.model.latent_dim), 3.0)})
    assert not torch.equal(base[child.name].change, moved[child.name].change)
    assert torch.equal(base[child.parent].change, moved[child.parent].change)


def test_correlation_and_tail():
    x = torch.randn(10, 6)
    assert abs(LI.corr(x, x) - 1) < 1e-6 and abs(LI.corr(x, -x) + 1) < 1e-6


def test_tail_reads_the_last_rows(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps({"i": i}) for i in range(1000)) + "\n", encoding="utf-8")
    rows = LI.tail_rows(p, 7)
    assert [r["i"] for r in rows] == list(range(993, 1000))


def test_code_centre_is_the_average_real_code():
    from mimo_bridge.tests.latent_body_fixtures import example
    learner = ShadowLearner(tree(), config())
    rows = []
    for step in range(1, 60):
        ex = example(step)
        rows.append(dict(history=ex.history.state_dict(), start=ex.start.state_dict(), end=ex.end.state_dict(),
                         executed=ex.executed, action_contract=ex.action_contract))
    rows = json.loads(json.dumps(rows, default=lambda t: t.tolist()))
    spread, count, offsets, centres = LI.code_scales(learner, rows)
    with torch.no_grad():
        codes = learner.model.encode(learner.batch([LI_example(r) for r in rows]))
    for name, z in codes.items():
        assert torch.allclose(centres[name], z.mean(0, keepdim=True), atol=1e-5)
        assert torch.allclose(spread[name], z.std(0).clamp_min(1e-6), atol=1e-5)


def LI_example(row):
    from lem.latent_body.trajectory import Example, Observation
    return Example(*(Observation.from_dict(row[k]) for k in ("history", "start", "end")),
                   torch.tensor(row["executed"]), row["action_contract"])


class _FakeSession:
    """A stand-in world: each handle moves half way to its ordered position in one step."""
    def __init__(self, names):
        from types import SimpleNamespace
        from lem.model.contact_head import PATCH_NAMES
        from mimo_bridge.tests.latent_body_fixtures import observation
        self.model = SimpleNamespace(handle_names=names)
        self.patches = [0] * len(PATCH_NAMES)
        self.bones = observation(0).bones.numpy().copy()
        self.t = 0.
        self.frames = [self._frame()]
        self.orders = []

    def _frame(self):
        from types import SimpleNamespace
        return SimpleNamespace(env_index=0, sim_time=self.t, bones=self.bones.copy(),
                               meta={"task": {"contact_patch_state": list(self.patches)}})

    def policy_act(self):
        pass

    def observe(self):
        self.t += .05
        self.frames = [self._frame()]

    def snapshot(self):
        return (self.bones.copy(), self.t)

    def hold_order(self):
        import numpy as np
        order = np.zeros((1, len(self.bones), 9), dtype=np.float32)
        order[0, :, :3], order[0, :, 3:7], order[0, :, 8] = self.bones[:, 4:7], self.bones[:, :4], .5
        return order

    def run_arm(self, snap, order, steps):
        bones, t = snap
        self.orders.append(order.copy())
        self.bones = bones.copy()
        self.bones[:, 4:7] += .5 * (order[0, :, :3] - bones[:, 4:7])
        self.t = t + .05 * steps
        self.frames = [self._frame()]


def test_every_trial_starts_from_the_centre_and_the_same_moment(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    learner = ShadowLearner(tree(), config())
    session = _FakeSession(learner.tree.handle_names)
    centres = {n.name: torch.full((1, learner.model.latent_dim), .3) for n in learner.tree.nodes}
    spread = {n.name: torch.ones(learner.model.latent_dim) for n in learner.tree.nodes}
    seen = []
    real = LI.decode_with
    monkeypatch.setattr(LI, "decode_with", lambda m, s, r, roots, over: seen.append((roots, over)) or real(m, s, r, roots, over))
    args = SimpleNamespace(warmup_steps=2, codes=4, centre="mean")
    out = LI.test_body(session, learner, spread, args, np.random.default_rng(0), centres)
    assert set(out) == {n.name for n in learner.tree.nodes}
    roots, over = seen[0]                                   # first node (an arm), first code
    assert all(torch.equal(v, centres[k]) for k, v in roots.items())
    child = [i for i, n in enumerate(learner.tree.nodes) if n.parent is not None][0]
    roots, over = seen[child * 4]
    assert all(torch.equal(v, centres[k]) for k, v in roots.items())
    assert torch.equal(over[learner.tree.nodes[child].name], centres[learner.tree.nodes[child].name])
    # the centre code's repeat reproduces its movement exactly in a deterministic world
    assert all(part["repeat_cm"] == 0 for part in out.values())
    args.centre = "zero"
    seen.clear()
    LI.test_body(_FakeSession(learner.tree.handle_names), learner, spread, args, np.random.default_rng(0), centres)
    assert all(float(v.abs().max()) == 0 for v in seen[0][0].values())
