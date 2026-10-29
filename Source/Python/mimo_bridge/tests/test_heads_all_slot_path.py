# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests of the all-slot paths beside the prediction heads (--heads-all-slot-path; lem/model/heads_all_slot_path.py;
docs/gain_of_function_v4/restart_candidate/RESTART_CANDIDATE_HEADS_2026-10-04.md).

Two groups. THE PATH ALONE (no learner): the zero start, the constant taken out without gradient, every slot reaching
the output through its own columns, the centre's update rule and when it moves, the first update's bound, frozen(),
the gauges, the head_read reading, its own random stream, the refusals. THE LEARNER (the small H84-shaped learner of
motor_asked_change_fixture.py with --motor-asked-change on, as production runs; CPU, synthetic frames; one shared run,
`learner_run`): off builds nothing; switching on at a checkpoint trained without the paths keeps the predictions, the
loss and the other weights' update; a checkpoint loads both ways and is refused where it must be; the planner's
descent leaves the paths' weights without gradient; an imagined row trains them as it trains the head beside them and
leaves the running centre where it is (version 2: the record's section 13); which lessons reach them; the liveness
monitor and repair cover their hidden layers; the stats block with the two fixed vectors; snapshots.

Planted defects: docs/gain_of_function_v4/prediction_heads_dead/planted_defects.py runs these tests on copies of the
code with one defect each."""
import copy
import gc
import json
import math
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from lem.model import heads_all_slot_path as ASP
import motor_asked_change_fixture as FX

torch.set_num_threads(1)          # beside production training: one thread

HEADS = ["vision", "tactile", "audio", "proprioception", "vestibular"]
S, D = 6, 32
PRODUCTION_FLAGS = ["--motor-asked-change", "on"]       # the candidate goes in on top of production's settings


# ------------------------------------------------------------------------------------------------ helpers
class Heads(nn.Module):
    """A stand-in for PredictionHeads with the real module's adding rule (lem/model/prediction_heads.py)."""

    def __init__(self):
        super().__init__()
        from lem.model.prediction_heads import PredictionHeads
        cfg = {"model": {"d_model": D, "residual_prediction": True,
                         "prediction_heads": {"n_heads": 4, "hidden_expansion": 1, "n_mlp_layers": 2}},
               "_prediction_head_modalities": HEADS}
        self.real = PredictionHeads(cfg)


def real_heads(seed=0):
    torch.manual_seed(seed)
    return Heads().real


def new_path(seed=0, random_output=False, constant=None, hidden=None):
    torch.manual_seed(seed)
    p = ASP.HeadsAllSlotPath(HEADS, S, D) if hidden is None else ASP.HeadsAllSlotPath(HEADS, S, D, hidden=hidden)
    if random_output:
        with torch.no_grad():
            for hp in p.paths.values():
                hp.net[2].weight.normal_(0, 0.1)
                hp.net[2].bias.normal_(0, 0.1)
    if constant is not None:
        p.constant_fn = lambda: constant
    p.train()
    return p


def rows(n=8, seed=0, constant=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, S, D, generator=g) * 0.5 + torch.randn(1, S, D, generator=g) * 5.0
    return x if constant is None else x + constant


def current(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(n, D, generator=g) for m in HEADS + ["motor"]}


# ------------------------------------------------------------------------------------------------ the path alone
def test_the_path_starts_at_zero_and_the_predictions_are_the_same_bits():
    heads = real_heads()
    x, cur = rows(), current()
    before = heads(x, current=cur, who="test")
    path = new_path()
    path.attach(heads)
    after = heads(x, current=cur, who="test")
    stacked, _d = heads.forward_stacked(x, current=cur, who="test")
    for i, m in enumerate(HEADS):
        assert torch.equal(after[m], before[m]) and after[m].dtype == before[m].dtype, m
        assert torch.equal(stacked[:, i], before[m]), m
    out = path(x)
    assert all(float(out[m].abs().max()) == 0.0 for m in HEADS)
    # the heads' own state dict and parameters do not change by the attachment
    assert not [k for k in heads.state_dict() if "all_slot" in k]
    assert not [n for n, _p in heads.named_parameters() if "all_slot" in n]
    path.detach()
    assert "all_slot_path" not in heads.__dict__


def test_the_output_is_added_to_every_head_in_both_forward_paths():
    heads = real_heads()
    x, cur = rows(), current()
    before = heads(x, current=cur, who="test")
    path = new_path(random_output=True)
    path.attach(heads)
    path.eval()
    want = path(x)
    after = heads(x, current=cur, who="test")
    stacked, d = heads.forward_stacked(x, current=cur, who="test")
    for i, m in enumerate(HEADS):
        assert float(want[m].abs().max()) > 0
        assert torch.allclose(after[m], before[m] + want[m], atol=1e-6), m
        assert torch.allclose(stacked[:, i], before[m] + want[m], atol=1e-6), m
        assert torch.equal(d[m], stacked[:, i])
    path.bypass = True
    assert all(torch.equal(heads(x, current=cur, who="test")[m], before[m]) for m in HEADS)


def test_the_memory_constant_is_taken_out_and_gets_no_gradient_from_the_subtraction():
    c = (torch.randn(S, D, generator=torch.Generator().manual_seed(3)) * 200.0).requires_grad_(True)
    path = new_path(random_output=True, constant=c)
    path.eval()
    plain = new_path(random_output=True)
    plain.load_state_dict(path.state_dict())
    plain.eval()
    x = rows()
    with_c = path(x + c.detach())
    without = plain(x)
    for m in HEADS:
        # the same numbers as without any constant, up to the float32 rounding of adding and subtracting 200
        assert torch.allclose(with_c[m], without[m], atol=2e-3), m
    sum(v.sum() for v in with_c.values()).backward()
    assert c.grad is None                              # the constant enters detached
    # a constant of that size left in would change the output
    path.constant_fn = None
    left_in = path(x + c.detach())
    assert max(float((left_in[m] - without[m]).abs().max()) for m in HEADS) > 0.05


def test_every_slot_reaches_the_output_through_its_own_columns():
    path = new_path(random_output=True)
    path.eval()
    x = rows()
    base = path(x)
    for j in range(S):
        xs = x.clone()
        xs[:, j] = x.roll(1, dims=0)[:, j]
        got = path(xs)
        for m in HEADS:
            assert float((got[m] - base[m]).abs().max()) > 1e-4, (j, m)
    for m in HEADS:
        W = path.paths[m].net[0].weight                # [hidden, S * rank]: every slot has its own columns
        assert path.rank == ASP.SLOT_RANK == 32 and path.paths[m].down.weight.shape == (path.rank, D)
        assert W.shape == (path.hidden, S * path.rank)
        blocks = W.reshape(path.hidden, S, path.rank).abs().sum(dim=(0, 2))
        assert bool((blocks > 0).all())


def test_the_centre_follows_the_training_calls_and_only_those():
    path = new_path()
    x = rows(seed=5)
    assert float(path.centre_weight) == 0.0
    path(x)                                            # a training call: gradient on, training mode
    assert torch.allclose(path.centre_mean, x.mean(0), atol=1e-5)      # the first call sets it
    w1 = float(path.centre_weight)
    assert abs(w1 - 1.0 / ASP.CENTRE_HORIZON_CALLS) < 1e-12
    y = rows(seed=6)
    path(y)
    a = 1.0 / ASP.CENTRE_HORIZON_CALLS
    w2 = (1 - a) * w1 + a
    want = x.mean(0) * ((1 - a) * w1 / w2) + y.mean(0) * (a / w2)
    assert torch.allclose(path.centre_mean, want, atol=1e-5)
    kept = path.centre_mean.clone()
    with torch.no_grad():
        path(rows(seed=7))                             # no gradient: a measurement or the lived pass
    with path.frozen():
        path(rows(seed=8))                             # a measurement pass with the weights detached
    with path.imagined():
        path(rows(seed=18))                            # an imagined row: the weights learn, the centre stays
    path.measuring_fn = lambda: True
    path(rows(seed=9))                                 # inside the liveness repair's unwatched block
    path.measuring_fn = None
    for p in path.parameters():
        p.requires_grad_(False)
    path(rows(seed=10))                                # the planner's descent (weights frozen)
    for p in path.parameters():
        p.requires_grad_(True)
    path.eval()
    path(rows(seed=11))                                # evaluation mode
    assert torch.equal(path.centre_mean, kept)
    # the centre is saved with the weights
    assert "centre_mean" in path.state_dict() and "centre_weight" in path.state_dict()


def test_the_input_is_centred_then_layernormed_so_a_shared_offset_does_not_reach_the_first_layer():
    path = new_path(random_output=True)
    x = rows(n=64, seed=12)
    for _ in range(3):
        path(x)                                        # the centre sits on these rows' mean
    path.eval()
    base = path(x)
    shared = torch.randn(1, S, D, generator=torch.Generator().manual_seed(13)) * 3.0
    with torch.no_grad():
        path.centre_mean.add_(shared[0])               # the centre has followed the offset
    moved = path(x + shared)
    for m in HEADS:
        assert torch.allclose(moved[m], base[m], atol=1e-4), m


def test_the_first_update_moves_the_output_by_at_most_the_gain_bound():
    path = new_path()
    x = rows(n=16, seed=14)
    path(x)
    opt = torch.optim.AdamW(path.parameters(), lr=3e-4, weight_decay=0.0)
    target = torch.randn(16, D, generator=torch.Generator().manual_seed(15))
    out = path(x)
    sum(((out[m] - target) ** 2).mean() for m in HEADS).backward()
    opt.step()
    path.eval()
    after = path(x)
    # Adam's first step is the learning rate per weight; the output layer reads `hidden` GELU outputs times the gain
    with torch.no_grad():
        xc = x - path.centre_mean
        for m in HEADS:
            p = path.paths[m]
            h = torch.nn.functional.gelu(p.net[0](p.down(p.norm(xc)).reshape(16, -1)))
            assert torch.equal(p.net[0](p.down(p.norm(xc)).reshape(16, -1)), path.hidden_input(m, x))
            bound = 3e-4 * (h.abs().sum(dim=1).max() + 1.0) * path.gain * 1.05
            assert 0.0 < float(after[m].abs().max()) <= float(bound), m
    assert path.gain == pytest.approx(1.0 / math.sqrt(D))


def test_inside_frozen_the_weights_get_no_gradient_and_the_row_keeps_its_own():
    path = new_path(random_output=True)
    x = rows(seed=16)
    path(x)
    x = x.clone().requires_grad_(True)
    free = path(x)
    with path.frozen():
        with path.frozen():                            # nests
            got = path(x)
        assert path.frozen_depth == 1
        for m in HEADS:
            assert torch.allclose(got[m], free[m], atol=1e-6), m
        sum(v.sum() for v in got.values()).backward()
    assert path.frozen_depth == 0
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert all(p.grad is None for p in path.parameters())
    out = path(x)
    sum(v.sum() for v in out.values()).backward()
    assert all(p.grad is not None for n, p in path.named_parameters())


def test_inside_imagined_the_weights_take_the_gradient_and_the_centre_stays():
    """Version 2 (the record's section 13): a lesson graded on an imagined row trains the path as it trains the head
    the path sits beside; the running centre is a mean over lived rows and does not move."""
    path = new_path(random_output=True)
    x = rows(seed=16)
    path(x)                                            # a training call sets the centre
    kept, w = path.centre_mean.clone(), float(path.centre_weight)
    with path.imagined():
        with path.imagined():                          # nests
            got = path(x)
        assert path.imagined_depth == 1 and path.frozen_depth == 0
        sum(v.sum() for v in got.values()).backward()
    assert path.imagined_depth == 0
    assert torch.equal(path.centre_mean, kept) and float(path.centre_weight) == w
    assert all(p.grad is not None for p in path.parameters())
    for m in HEADS:
        assert float(path.paths[m].net[2].weight.grad.abs().sum()) > 0, m
        assert float(path.paths[m].down.weight.grad.abs().sum()) > 0, m
    # the rule and its stamp
    assert ASP.IMAGINED_ROWS == "learn" and path.imagined_rows == "learn"
    assert path.stamp()["imagined_rows"] == "learn" and path.stamp()["version"] == "heads_all_slot_path_v2"
    # version 1's rule, kept behind the switch: inside imagined() the weights are detached
    old = new_path(random_output=True)
    old.imagined_rows = "detached"
    old(x)
    xg = x.clone().requires_grad_(True)
    with old.imagined():
        assert old.frozen_depth == 1 and old.imagined_depth == 0
        sum(v.sum() for v in old(xg).values()).backward()
    assert old.frozen_depth == 0 and xg.grad is not None
    assert all(p.grad is None for p in old.parameters())


def test_a_call_without_gradient_earlier_in_one_autocast_block_does_not_cut_the_weights_gradient():
    """Inside one autocast block a weight first used without gradient is copied to the 16-bit format once and every
    later use in the block reads that copy, which carries no gradient (measured on the graphics card Oct 4, 2026:
    D:/UEMA_Research/external/D/UEMA_scratch/vestibular_token2/logs/live_bench_debug3.log; LemLite.step is one autocast block). The path
    computes in float32 outside autocast, so its weights are never copied: a measurement pass earlier in the block
    cannot cut a later lesson's gradient. The control layer shows this build of torch has the effect on the CPU."""
    path = new_path(random_output=True)
    x = rows(seed=19)
    control = nn.Linear(D, D)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            path(x)
            control(x[:, 0])
        out = path(x)
        loss = sum(v.float().sum() for v in out.values()) + control(x[:, 0]).float().sum()
    loss.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in path.parameters())
    assert all(v.dtype == torch.float32 for v in out.values())
    assert control.weight.grad is None                 # the control: a layer computed under autocast lost it


def test_fixed_parts_reads_the_two_fixed_vectors_and_their_cosine():
    heads = real_heads(seed=3)
    path = new_path(seed=4)
    path.attach(heads)
    x = rows(n=16, seed=21)
    g = torch.Generator().manual_seed(22)
    cur = {m: torch.randn(16, D, generator=g) for m in HEADS}
    with torch.no_grad():
        path.bypass = True
        old = heads(x, current=cur, who="test")
        path.bypass = False
        want = {m: (old[m] - cur[m]).mean(0) for m in HEADS}
        # the path's output layer made to add minus the old head's fixed vector to every row of the tactile head,
        # and the old head's fixed vector itself to every row of the audio head
        path.paths["tactile"].net[2].bias.copy_(-want["tactile"] / path.gain)
        path.paths["audio"].net[2].bias.copy_(want["audio"] / path.gain)
    kept = (path.centre_mean.clone(), float(path.centre_weight))
    got = ASP.fixed_parts(heads, x, cur)
    assert set(got) == set(HEADS)
    rms = lambda t: float(t.pow(2).mean().sqrt())
    for m in HEADS:
        assert got[m]["old_rms"] == pytest.approx(rms(want[m]), rel=1e-4) and got[m]["rows"] == 16
    t, a_, v = got["tactile"], got["audio"], got["vision"]
    assert t["path_rms"] == pytest.approx(t["old_rms"], rel=1e-4) and t["cosine"] == pytest.approx(-1.0, abs=1e-3)
    assert t["sum_rms"] < 1e-5
    assert a_["cosine"] == pytest.approx(1.0, abs=1e-3) and a_["sum_rms"] == pytest.approx(2 * a_["old_rms"], rel=1e-4)
    assert v["path_rms"] == 0.0 and v["cosine"] is None and v["sum_rms"] == pytest.approx(v["old_rms"], rel=1e-6)
    # a reading: the bypass is off again, no centre moved, no gradient was written
    assert path.bypass is False and torch.equal(path.centre_mean, kept[0]) and float(path.centre_weight) == kept[1]
    assert all(p.grad is None for p in path.parameters())
    path.detach()


def test_the_windows_gauges_read_the_armed_call_and_report_reach_and_liveness():
    path = new_path()
    x = rows(n=8, seed=17)
    path(x)
    s = path.window_summary()
    assert s["rows"] == 0 and s["heads"]["tactile"]["output_weight_rms"] == 0.0     # nothing armed
    path.arm_gauge()
    path(x)
    assert path.gauge_armed is False and len(path.gauge_calls) == 1
    path(rows(seed=18))                                # not armed: not kept
    assert len(path.gauge_calls) == 1
    s = path.window_summary()
    assert s["rows"] == 8 and len(s["input_rms"]) == S and len(s["centre_lag"]) == S
    assert all(h["output_rms"] == 0.0 and "slot_reach" not in h for h in s["heads"].values())
    assert all(h["live"] == 1.0 for h in s["heads"].values())
    assert len(path.gauge_calls) == 0                  # a fresh window
    with torch.no_grad():
        for hp in path.paths.values():
            hp.net[2].weight.normal_(0, 0.1)
        path.paths["tactile"].net[0].weight[:, path.rank:] = 0.0  # the tactile path reads slot 0 only
    path.arm_gauge()
    path(x)
    s = path.window_summary()
    assert s["heads"]["vision"]["slot_reach"]["spread"] > 0.8
    t = s["heads"]["tactile"]["slot_reach"]
    assert t["spread"] < 0.05 and t["largest_share"] > 0.99 and t["shares"][0] > 0.99
    with torch.no_grad():
        path.paths["audio"].net[0].bias[::2] = -30.0
    path.arm_gauge()
    path(x)
    assert path.window_summary()["heads"]["audio"]["live"] == 0.5


def test_head_read_counts_the_pool_and_the_path_together():
    heads = real_heads()
    x, cur = rows(n=16, seed=19), current(n=16)
    with torch.no_grad():
        only = ASP.head_read(heads, x, cur)
    assert set(only) == set(HEADS)
    path = new_path(random_output=True)
    path.eval()
    with torch.no_grad():
        for hp in path.paths.values():
            hp.net[2].weight.mul_(5000.0)              # the path dominates the small random head
        path.paths["tactile"].net[0].weight[:, path.rank:] = 0.0
    path.attach(heads)
    with torch.no_grad():
        both = ASP.head_read(heads, x, cur)
    assert both["tactile"]["largest_share"] > 0.9 and both["tactile"]["spread"] < 0.3
    assert both["vision"]["spread"] > 0.8 and both["vision"]["rows"] == 16
    # a head whose prediction answers to no slot reads spread 0 with no shares
    class Dead(nn.Module):
        head_modalities = HEADS

        def forward(self, x, current=None, who=""):
            return {m: current[m] for m in HEADS}
    dead = ASP.head_read(Dead(), x, cur)
    assert dead["vision"]["spread"] == 0.0 and dead["vision"]["shares"] is None


def test_the_path_is_built_in_its_own_random_stream():
    torch.manual_seed(77)
    a = torch.get_rng_state().clone()
    p1 = ASP.build(HEADS, S, D, seed=0)
    assert torch.equal(torch.get_rng_state(), a)       # the run's stream is where it was
    torch.manual_seed(78)
    p2 = ASP.build(HEADS, S, D, seed=0)
    for (n1, v1), (_n2, v2) in zip(p1.state_dict().items(), p2.state_dict().items()):
        assert torch.equal(v1, v2), n1
    p3 = ASP.build(HEADS, S, D, seed=1)
    assert not torch.equal(p3.paths["vision"].net[0].weight, p1.paths["vision"].net[0].weight)
    for hp in p1.paths.values():
        assert float(hp.net[2].weight.abs().max()) == 0.0 and float(hp.net[2].bias.abs().max()) == 0.0


def test_refusals_of_the_path():
    path = new_path()
    with pytest.raises(ValueError, match="6 slots"):
        path(torch.zeros(2, 5, D))
    heads = real_heads()
    path.attach(heads)
    with pytest.raises(RuntimeError, match="attached already"):
        path.attach(heads)
    other = ASP.HeadsAllSlotPath(["vision"], S, D)
    with pytest.raises(ValueError, match="built for heads"):
        other.attach(real_heads())
    from mimo_bridge.lem_model import LemLite
    with pytest.raises(ValueError, match="--heads-all-slot-path must be"):
        LemLite(device="cpu", heads_all_slot_path="maybe")


# ------------------------------------------------------------------------------------------------ the learner
def _built(tmp, name, flag, ckpt=None):
    """A learner built as the trainer builds it, with its checkpoint loaded as the trainer loads it. flag None = the
    launch line without the flag."""
    from mimo_bridge import train_mimo as T
    extra = list(PRODUCTION_FLAGS) + ([] if flag is None else ["--heads-all-slot-path", flag])
    args = T.parse_args(FX.argv(out=str(tmp / name), extra=extra))
    args.n_envs = int(args.n_envs)
    model = T.build_learner(args, str(tmp / name / "checkpoints"))
    loaded = None if ckpt is None else model.load_checkpoint(str(ckpt))
    model.reset_context(args.n_envs)
    model.enable_replay(args.n_envs, size_per_env=64)
    model.replay_batch = 4
    return model, args, loaded


def _seed(model, seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.noise_rng.manual_seed(seed)


LAYERS = [ASP.liveness_layer(m) for m in HEADS]


def _facts(model, args):
    from mimo_bridge import train_mimo as T
    lv = model.liveness
    path = getattr(model, "heads_all_slot_path", None)
    return dict(has_path=path is not None, has_attribute=hasattr(model, "heads_all_slot_path"),
                receipt="heads_all_slot_path" in model.receipt,
                receipt_params=model.receipt["component_params"].get("heads_all_slot_path"),
                side="heads_all_slot_path" in model._side_modules(),
                liveness_layers=list(lv.layer_names), liveness_rows=sorted(lv.rows.keys()),
                n_params=int(model.n_params),
                acting=T.acting_configuration(args).get("heads_all_slot_path", "absent"),
                frozen_in_planning=any(m is path for m in model.intention_gen._wellness_freeze),
                last_params=(None if path is None else
                             all(a is b for a, b in zip(model.params[-len(list(path.parameters())):],
                                                        path.parameters()))),
                optimizer_has_path=(None if path is None else
                                    all(any(p is q for g in model.optimizer.param_groups for q in g["params"])
                                        for p in path.parameters())))


class LessonProbe:
    """Per lesson term: the summed absolute gradient that reaches the paths' weights (LemLite.lesson_probe)."""

    def __init__(self, params):
        self.terms, self.params = {}, params
        self.reach, self.seen = {}, {}

    def before_backward(self, model, loss):
        for k, v in self.terms.items():
            self.seen[k] = self.seen.get(k, 0) + 1
            if not torch.is_tensor(v) or not v.requires_grad:
                continue
            g = torch.autograd.grad(v, self.params, retain_graph=True, allow_unused=True)
            self.reach[k] = self.reach.get(k, 0.0) + sum(float(x.abs().sum()) for x in g if x is not None)

    def after_clip(self, model, grad_norm):
        pass


class PathCalls:
    """Every call of the path while attached: its call site (the first of PRIORITY among the calling stack's function
    names, else "transition", else "step"), whether gradients were enabled, whether its weights required a gradient,
    whether it was frozen, and whether the call moved the running centre."""

    PRIORITY = ("switch_check", "head_read", "_swap_test_arms", "replay_update", "_run_imagination_compiled",
                "generate_action", "_causal_score_candidates", "_causal_plan_record", "_plan_chain",
                "_latent_for_order", "predict_before_execution")

    def __init__(self, model):
        self.calls, self.model = [], model
        self.path = model.heads_all_slot_path
        self._orig = self.path.forward
        path = self.path

        def forward(last_step):
            import sys
            names = []
            f = sys._getframe(1)
            while f is not None and len(names) < 80:
                names.append(f.f_code.co_name)
                f = f.f_back
            site = next((n for n in self.PRIORITY if n in names), None)
            if site is None:
                site = "transition" if "transition" in names else ("step" if "step" in names else "other")
            w0 = float(path.centre_weight)
            out = self._orig(last_step)
            self.calls.append(dict(site=site, grad=bool(torch.is_grad_enabled()),
                                   trains=bool(path.paths["vision"].net[0].weight.requires_grad),
                                   frozen=bool(path.frozen_depth), imagined=bool(path.imagined_depth),
                                   centre_moved=float(path.centre_weight) != w0,
                                   rows=int(last_step.shape[0])))
            return out
        path.forward = forward

    def remove(self):
        self.path.forward = self._orig


def _gradient_row_by_row(model, gauge):
    """The lesson's gradient on the heads' input, one row at a time (each row's own mean squared error, each row's own
    derivative): the reference the stats block's lesson_gradient must equal."""
    from lem.model import liveness as LV
    heads = getattr(model.prediction_heads, "_orig_mod", model.prediction_heads)
    sq, n = {}, {}
    with LV.unwatched(model), model.heads_all_slot_path.frozen(), torch.enable_grad(), \
            torch.amp.autocast("cuda", enabled=model.use_amp, dtype=model.dtype):
        for x, cur, tgt in gauge:
            for i in range(x.shape[0]):
                xi = x[i:i + 1].clone().requires_grad_(True)
                preds = heads(xi, current={m: v[i:i + 1] for m, v in cur.items()}, who="test")
                for m, t in tgt.items():
                    loss = (preds[m].float() - t[i:i + 1].float()).pow(2).mean()
                    g, = torch.autograd.grad(loss, xi, retain_graph=True)
                    sq[m] = sq.get(m, 0.0) + float(g.float().pow(2).sum())
                    n[m] = n.get(m, 0) + g.numel()
    return {m: (sq[m] / n[m]) ** 0.5 for m in sq}


@pytest.fixture(scope="module")
def learner_run(tmp_path_factory):
    """One shared run: a learner trained without the paths (with the asked-change path, as production) for 12 model
    steps and saved; that checkpoint resumed without the paths and with them (the switch), 6 model steps each on the
    same frames and random streams; the switched learner trained on to model step 60 with every call of the path
    recorded, saved, and read with a lesson probe for 12 more; its checkpoint loaded into a build with the paths."""
    from mimo_bridge import copied_execution as CE
    from mimo_bridge import continuation_state as CS
    tmp = tmp_path_factory.mktemp("all_slot_path")
    R = {}
    off0, args_off, _ = _built(tmp, "off0", None)
    R["off"] = _facts(off0, args_off)
    world = FX.SyntheticWorld(args_off.n_envs, seed=0)
    rows_ = FX.drive(off0, world, 12, replay_every=2)
    off0.step_count = 12
    ckpt0 = off0.save_checkpoint(12)
    carry = rows_[-1]["carry"]
    R["ckpt0"] = ckpt0
    del off0
    gc.collect()

    def continued(model, n, start, state, **kw):
        w, c = copy.deepcopy((world, state))
        _seed(model)
        return FX.drive(model, w, n, start=start, state=c, digest_every=1, replay_every=2,
                        digest_skip=("heads_all_slot_path",) + tuple(LAYERS), **kw), w

    off1, _a, loaded = _built(tmp, "off1", None, ckpt0)
    R["rows_off"], _w = continued(off1, 6, 12, carry)
    del off1
    gc.collect()
    on1, args_on, loaded = _built(tmp, "on1", "on", ckpt0)
    R["on"] = _facts(on1, args_on)
    R["switch"] = copy.deepcopy(on1.heads_all_slot_path_switch)
    R["switch_stamp"] = copy.deepcopy(on1.receipt["heads_all_slot_path"])
    R["loaded_step"] = loaded["step"]
    R["centre_weight_after_load"] = float(on1.heads_all_slot_path.centre_weight)
    calls = PathCalls(on1)
    weight_max = {}

    def before(step, prev_exec):
        weight_max[step] = max(float(p.net[2].weight.abs().max()) for p in on1.heads_all_slot_path.paths.values())
    R["rows_on"], w_on = continued(on1, 6, 12, carry, before_step=before)
    more = FX.drive(on1, w_on, 42, start=18, state=R["rows_on"][-1]["carry"], replay_every=2, before_step=before)
    calls.remove()
    R.update(calls=calls.calls, weight_max=weight_max)
    snap = CE.capture(on1)
    R["capture_has_module"] = "heads_all_slot_path" in snap["modules"]
    R["companion_names"] = sorted(CS.capture_learner(on1)["data"].keys())
    watched = {k: int(r.seen_n) for k, r in on1.liveness.rows.items()}
    gauge = [(c[0].clone(), {m: v.clone() for m, v in c[1].items()}, {m: v.clone() for m, v in c[2].items()})
             for c in on1.heads_all_slot_path_gauge]
    grads = [None if p.grad is None else p.grad.detach().clone() for p in on1.params]
    centre = (on1.heads_all_slot_path.centre_mean.clone(), float(on1.heads_all_slot_path.centre_weight))
    rng = FX.sha(torch.get_rng_state())
    R["report"] = on1.heads_all_slot_path_report()
    R["report_left_every_grad_the_centre_and_the_random_stream"] = (
        all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g, p.grad))
            for g, p in zip(grads, on1.params))
        and torch.equal(centre[0], on1.heads_all_slot_path.centre_mean)
        and centre[1] == float(on1.heads_all_slot_path.centre_weight) and rng == FX.sha(torch.get_rng_state()))
    R["gauge_calls"], R["gauge_rows"] = len(gauge), sum(int(c[0].shape[0]) for c in gauge)
    R["gradient_row_by_row"] = _gradient_row_by_row(on1, gauge)
    R["report_joined_no_watching_window"] = watched == {k: int(r.seen_n) for k, r in on1.liveness.rows.items()}
    R["watched_rows_before_the_report"] = sum(watched.values())
    R["report_again"] = on1.heads_all_slot_path_report()
    R["liveness_summary"] = on1.liveness.window_summary(on1)
    on1.step_count = 60
    ckpt1 = on1.save_checkpoint(60)
    R["ckpt1"] = ckpt1
    path = on1.heads_all_slot_path
    R["path_state"] = {k: v.detach().clone() for k, v in path.state_dict().items()}
    R["path_adam"] = {n: {k: v.detach().clone() for k, v in on1.optimizer.state[p].items() if torch.is_tensor(v)}
                      for n, p in path.named_parameters() if p in on1.optimizer.state}
    probe = LessonProbe([p.net[2].weight for p in path.paths.values()])
    on1.lesson_probe = probe
    on1.loss_manager.term_probe = probe.terms
    FX.drive(on1, w_on, 12, start=60, state=more[-1]["carry"], replay_every=0,
             on_step=lambda step, model, res: probe.terms.clear())
    R["reach"], R["seen"] = dict(probe.reach), dict(probe.seen)
    del on1
    gc.collect()
    on2, _a, loaded = _built(tmp, "on2", "on", ckpt1)
    R["on2_state"] = {k: v.detach().clone() for k, v in on2.heads_all_slot_path.state_dict().items()}
    R["on2_adam"] = {n: {k: v.detach().clone() for k, v in on2.optimizer.state[p].items() if torch.is_tensor(v)}
                     for n, p in on2.heads_all_slot_path.named_parameters() if p in on2.optimizer.state}
    R["on2_switch"] = on2.heads_all_slot_path_switch
    R["on2_stamp"] = copy.deepcopy(on2.receipt["heads_all_slot_path"])
    R["on2_step"] = loaded["step"]
    del on2
    gc.collect()
    R["tmp"] = tmp
    return R


def test_off_builds_nothing_and_on_builds_the_paths(learner_run):
    off, on = learner_run["off"], learner_run["on"]
    assert off["has_attribute"] is False and off["receipt"] is False and off["side"] is False
    assert not [n for n in off["liveness_layers"] if "all_slot" in n]
    assert off["acting"] == "absent"                 # the acting fingerprint of a launch without the paths is unchanged
    assert on["has_path"] and on["receipt"] and on["side"] and on["acting"] == "on"
    assert on["optimizer_has_path"] and on["frozen_in_planning"] and on["last_params"]
    assert on["liveness_layers"] == off["liveness_layers"] + LAYERS
    rank = min(ASP.SLOT_RANK, D)                    # the LayerNorm, the slot matrix, the hidden layer, the output layer
    per_head = (2 * D) + (rank * D) + (S * rank * D + D) + (D * D + D)
    assert on["receipt_params"] == per_head * len(HEADS)
    new_rows = [k for k in on["liveness_rows"] if k not in off["liveness_rows"]]
    assert sorted(new_rows) == sorted(n.replace(".", "__") for n in LAYERS)


def test_switching_on_at_a_checkpoint_keeps_predictions_loss_and_the_other_weights_update(learner_run):
    R = learner_run
    sw = R["switch"]
    assert sw is not None and sw["exact"] and sw["float32"]["bit_identical"] and sw["float32"]["change_max_abs"] == 0.0
    assert sw["output_weight_max_abs"] == 0.0 and sw["output_bias_max_abs"] == 0.0 and sw["rows"] == 64
    assert sw["at_step"] == 12 and sw["version"] == ASP.ALL_SLOT_PATH_VERSION and R["loaded_step"] == 12
    assert R["switch_stamp"]["switch"]["at_step"] == 12
    assert R["centre_weight_after_load"] == 0.0        # the switch check moved no centre
    off, on = R["rows_off"], R["rows_on"]
    same = ("loss", "grad", "command", "gaze", "breakdown", "wellness", "replay", "torch_rng", "numpy_rng",
            "noise_rng", "tokens", "predictions", "state")
    # model step 12: every output layer is zero through the whole model step's forward passes
    assert R["weight_max"][12] == 0.0
    for k in same:
        assert on[0][k] == off[0][k], (12, k)
    # and the comparison can see the paths: once an output layer has moved, training differs
    assert max(R["weight_max"][s] for s in range(13, 18)) > 0.0
    assert any(on[i]["loss"] != off[i]["loss"] for i in range(1, 6))


def test_a_checkpoint_loads_both_ways_and_is_refused_where_it_must_be(learner_run):
    R = learner_run
    tmp = R["tmp"]
    assert R["on2_step"] == 60 and R["on2_switch"] is None        # no second switch
    assert set(R["on2_state"]) == set(R["path_state"])
    for k, v in R["path_state"].items():
        assert torch.equal(R["on2_state"][k], v), k
    assert float(R["path_state"]["paths.tactile.net.2.weight"].abs().max()) > 0.0
    assert float(R["path_state"]["centre_weight"]) > 0.0           # the centre is saved with the weights
    assert set(R["on2_adam"]) == set(R["path_adam"]) and R["path_adam"]
    for n, st in R["path_adam"].items():
        for k, v in st.items():
            assert torch.equal(R["on2_adam"][n][k], v), (n, k)
    assert R["on2_stamp"]["switch"] == R["switch_stamp"]["switch"]
    off2, _a, _l = _built(tmp, "off2", None)
    with pytest.raises(ValueError, match="--heads-all-slot-path on"):
        off2.load_checkpoint(R["ckpt1"])
    del off2
    gc.collect()
    other = tmp / "other_version" / os.path.basename(R["ckpt1"])
    shutil.copytree(R["ckpt1"], other)
    receipt = json.loads((other / "model_receipt.json").read_text(encoding="utf-8"))
    receipt["heads_all_slot_path"]["version"] = "heads_all_slot_path_v0"
    (other / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    on3, _a, _l = _built(tmp, "on3", "on")
    with pytest.raises(ValueError, match="heads_all_slot_path_v0"):
        on3.load_checkpoint(str(other))
    import lem.training.checkpoint as CK
    real = CK.CheckpointManager.load

    def keep_planted(self, *a, **kw):
        out = real(self, *a, **kw)
        with torch.no_grad():
            on3.heads_all_slot_path.paths["tactile"].net[2].weight.fill_(0.01)
        return out
    CK.CheckpointManager.load = keep_planted
    try:
        with pytest.raises(RuntimeError, match="output\\s+layer is not zero"):
            on3.load_checkpoint(R["ckpt0"])
    finally:
        CK.CheckpointManager.load = real
    del on3
    gc.collect()


def test_imagined_rows_train_the_weights_leave_the_centre_and_planning_trains_nothing(learner_run):
    calls = learner_run["calls"]
    sites = {c["site"] for c in calls}
    assert "step" in sites and "replay_update" in sites
    imagined = ("_run_imagination_compiled", "generate_action", "_causal_score_candidates", "_causal_plan_record",
                "_plan_chain", "_latent_for_order", "predict_before_execution", "transition")
    assert sites & set(imagined), sites
    for c in calls:
        if c["site"] in imagined:
            # an imagined row (version 2): inside imagined(), never detached, the centre never moves
            assert c["imagined"] and not c["frozen"] and not c["centre_moved"], c
        elif c["site"] in ("head_read", "switch_check", "_swap_test_arms"):
            assert not c["grad"] and not c["centre_moved"], c
        else:
            assert c["site"] in ("step", "replay_update"), c["site"]
            assert not c["frozen"] and not c["imagined"], c
            assert c["centre_moved"] == (c["grad"] and c["trains"]), c
    # the imagination chain's rows train the paths as they train the old heads
    chain = [c for c in calls if c["site"] == "_run_imagination_compiled"]
    assert chain and all(c["grad"] and c["trains"] for c in chain)
    # the planner's descent trains nothing: the intention generator freezes the paths' weights with the heads'
    planning = [c for c in calls if c["site"] == "generate_action"]
    assert planning and all(c["grad"] and not c["trains"] for c in planning)
    assert any(c["centre_moved"] for c in calls if c["site"] == "step")
    assert any(c["centre_moved"] for c in calls if c["site"] == "replay_update")


def test_which_lessons_reach_the_paths(learner_run):
    reach, seen = learner_run["reach"], learner_run["seen"]
    for k in ("next_frame.proprioception", "next_frame.tactile", "next_frame.vision", "next_frame.audio",
              "next_frame.vestibular"):
        assert seen.get(k, 0) > 0 and reach.get(k, 0.0) > 0.0, k
    # the imagination chain's loss reaches the paths' weights (version 2: the lesson that trains the old head trains
    # the path beside it; with the weights detached there, as in version 1, this reads 0.0)
    assert seen.get("imagination", 0) > 0 and reach.get("imagination", 0.0) > 0.0


def r_rows(s, m):
    return s["head_read"][m]["rows"]


def test_the_stats_block_reports_the_paths_and_head_read(learner_run):
    s = learner_run["report"]
    assert s["version"] == ASP.ALL_SLOT_PATH_VERSION and s["rows"] > 0 and 0 < s["centre_weight"] <= 1
    assert len(s["input_rms"]) == S and len(s["moment_rms"]) == S and len(s["centre_lag"]) == S
    for m in HEADS:
        h = s["heads"][m]
        assert h["output_weight_rms"] > 0 and h["output_rms"] > 0 and h["live"] > 0.9
        assert 0.0 <= h["output_fixed_rms"] <= h["output_rms"] * (1 + 1e-6)
        f = s["fixed"][m]
        assert f["old_rms"] > 0 and f["path_rms"] > 0 and f["sum_rms"] >= 0 and -1.0 <= f["cosine"] <= 1.0
        assert f["rows"] == r_rows(s, m)
        assert 0.0 <= h["slot_reach"]["spread"] <= 1.0 and len(h["slot_reach"]["shares"]) == S
        r = s["head_read"][m]
        assert 0.0 < r["spread"] <= 1.0 and r["rows"] >= 2
    # head_read ran the heads without a row joining any of the liveness repair's watching windows
    assert learner_run["report_joined_no_watching_window"] and learner_run["watched_rows_before_the_report"] > 0
    again = learner_run["report_again"]
    assert again["rows"] == 0 and "head_read" not in again and "lesson_gradient" not in again     # a fresh window
    assert "fixed" not in again
    # the lesson's gradient on the transformer's output: every head, the lesson rows of the kept calls, equal to the
    # row-by-row reference; computing it wrote no parameter's .grad, moved no centre and drew no random number
    lg, ref = s["lesson_gradient"], learner_run["gradient_row_by_row"]
    assert set(lg) == set(HEADS) == set(ref) and learner_run["gauge_calls"] >= 1
    for m in HEADS:
        assert lg[m]["rows"] == learner_run["gauge_rows"] and lg[m]["rms"] > 0.0
        assert lg[m]["rms"] == pytest.approx(ref[m], rel=1e-4), m
    assert learner_run["report_left_every_grad_the_centre_and_the_random_stream"]
    lv = learner_run["liveness_summary"]
    for n in LAYERS:
        assert n in lv["layers"] and n in lv["watching"]


def test_snapshots_carry_the_paths_as_a_module_and_never_as_episode_state(learner_run):
    R = learner_run
    assert R["capture_has_module"]
    assert not [n for n in R["companion_names"] if "all_slot" in n or "asp" in n]


def _fresh_build_marks(model):
    """Of a learner just built: the random stream's state, and one hash over every weight a build with the paths
    shares with a build without them."""
    from lem.model.efference import _named_components
    shared = []
    for comp, mod in sorted(_named_components(model).items()):
        if comp == "heads_all_slot_path":
            continue
        for n, p in sorted(mod.named_parameters()):
            if "heads_all_slot_path" not in n:
                shared.append(p.detach().float().reshape(-1))
    return FX.sha(torch.get_rng_state()), FX.sha(torch.cat(shared))


def test_the_paths_gradient_is_clipped_in_a_group_of_its_own(tmp_path):
    """A gradient of the paths far above the clip does not shrink the other weights' gradient (lem_model.py
    _clip_training_gradients: the paths are a clip group of their own) and is itself brought to the clip; the norm
    the model step reports is the other weights'. (The switch test cannot see this: the first model step after a
    resume runs no lesson through the heads, so the paths' gradient is zero there.)"""
    model, _a, _l = _built(tmp_path, "clip", "on")
    paths = list(model.heads_all_slot_path.parameters())
    ids = {id(p) for p in paths}
    others = [p for p in model.params if id(p) not in ids]
    for p in model.params:
        p.grad = torch.zeros_like(p)
    others[0].grad.fill_(1e-4)
    kept = others[0].grad.clone()
    assert float(kept.norm()) < model.grad_clip
    for p in paths:
        p.grad = torch.full_like(p, 10.0)
    norm = model._clip_training_gradients()
    assert torch.equal(others[0].grad, kept)
    assert norm == pytest.approx(float(kept.norm()), rel=1e-5)
    total = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in paths))
    assert total == pytest.approx(float(model.grad_clip), rel=1e-3)
    del model
    gc.collect()


def test_the_liveness_repair_revives_a_paths_hidden_layer(tmp_path):
    off, _a, _l = _built(tmp_path, "lv_off", None)
    marks_off = _fresh_build_marks(off)
    del off
    gc.collect()
    model, args, _l = _built(tmp_path, "lv", "on")
    # a fresh build with the paths starts from the same weights everywhere else and leaves the run's random stream
    # where a build without them leaves it
    assert _fresh_build_marks(model) == marks_off
    name = ASP.liveness_layer("tactile")
    lin = model.heads_all_slot_path.paths["tactile"].net[0]
    with torch.no_grad():
        lin.bias[::2] = -30.0
    world = FX.SyntheticWorld(args.n_envs, seed=2)
    model.liveness.revive_min_samples = 10 ** 9
    rows_ = FX.drive(model, world, 10, digest_every=0)
    monitor = model.liveness.window_summary(model)
    own = model.heads_all_slot_path_report()
    assert monitor["layers"][name] == 0.5 and own["heads"]["tactile"]["live"] == 0.5
    model.liveness.revive_min_samples = 60
    FX.drive(model, world, 20, start=10, state=rows_[-1]["carry"], digest_every=0)
    monitor = model.liveness.window_summary(model)
    assert name in FX.revivals(model)
    assert monitor["revived_units"][name] == D // 2 and monitor["layers"][name] == 1.0


def test_the_trainers_flag():
    from mimo_bridge import train_mimo as T
    base = ["--out", "x", "--device", "cpu"]
    assert T.parse_args(base).heads_all_slot_path == "off"
    assert T.parse_args(base + ["--heads-all-slot-path", "on"]).heads_all_slot_path == "on"
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--heads-all-slot-path", "maybe"])
    import inspect
    src = inspect.getsource(T)
    assert '"heads_all_slot_path": model.heads_all_slot_path_report()' in src      # the stats block
    assert '"type": "heads_all_slot_path_switch"' in src                           # the switch record
    old = T.parse_args(base)
    del old.heads_all_slot_path
    assert "heads_all_slot_path" not in T.acting_configuration(old)
    assert "heads_all_slot_path" not in T.acting_configuration(T.parse_args(base))
    assert T.acting_configuration(T.parse_args(base + ["--heads-all-slot-path", "on"]))["heads_all_slot_path"] == "on"
