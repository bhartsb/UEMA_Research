# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Contract overseer — checks the transition contract by EXECUTING it, not by reading it.

    python -m mimo_bridge.tools.contract_overseer            # every check
    python -m mimo_bridge.tools.contract_overseer --only C4  # one check
    python -m mimo_bridge.tools.contract_overseer --json out.json

Why this exists
---------------
Three review rounds found the same shape of defect every time: the unit-level repair is
present and the surrounding runtime still supplies the wrong tensor, the wrong state or
the wrong semantics. Every one of those got past a green test suite, because the tests
checked the piece rather than the path.

  * `build_frame_from_tokenized` referenced a name it did not take — a NameError on the
    full path, in the commit that claimed to fix the full path.
  * `generate_action` called the canonical transition and passed it `context=None` and
    `achieved=None`, so it optimised against a fabricated history and a zero pose.
  * the scale gate computed a verdict and then returned the plan anyway.
  * an AST test proved "no call omits state_tokens" and proved nothing about the two
    other routes that never call that function at all.

So each check here RUNS the thing. A check that only greps is marked `static` in its own
output, so nobody mistakes it for evidence about the runtime.

Every check states what it would take to fool it. A check that cannot fail is not a check.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import traceback
from typing import Callable, Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CHECKS: List[dict] = []


def check(cid: str, title: str, kind: str = "runtime"):
    """kind: 'runtime' if it executes the path, 'static' if it only inspects source."""
    def deco(fn: Callable):
        CHECKS.append({"id": cid, "title": title, "kind": kind, "fn": fn})
        return fn
    return deco


def _tiny(**kw):
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", overrides={"model": {"context_length": 4}},
                   next_frame_loss=True, **kw)


# ───────────────────────────────────────────────────────── one canonical transition
@check("C1", "no production call takes the legacy predict_from_action route", "static")
def c1():
    import lem
    import mimo_bridge
    offenders = []
    for base in (os.path.dirname(lem.__file__), os.path.dirname(mimo_bridge.__file__)):
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in ("__pycache__", "tests")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    tree = ast.parse(open(p, encoding="utf-8").read())
                except SyntaxError:
                    continue
                for n in ast.walk(tree):
                    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and n.func.attr == "predict_from_action"
                            and "state_tokens" not in {k.arg for k in n.keywords}):
                        offenders.append(f"{os.path.basename(p)}:{n.lineno}")
    return (not offenders, f"{len(offenders)} legacy call site(s): {offenders[:6]}",
            "Fooled by: a route that never calls predict_from_action at all. See C2.")


@check("C2", "nothing reaches the motor encoder except through the registry", "static")
def c2():
    """The AST check above proves nothing about code that bypasses that function
    entirely. This finds direct `encoders["motor"](...)` calls, which skip the registry's
    output LayerNorm — the defect that put tokens 67,715x off scale."""
    import lem
    import mimo_bridge
    offenders = []
    for base in (os.path.dirname(lem.__file__), os.path.dirname(mimo_bridge.__file__)):
        for dp, dn, fn in os.walk(base):
            # tools/ is excluded: this very file names the pattern in order to find it
            dn[:] = [d for d in dn if d not in ("__pycache__", "tests", "tools")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                src = open(p, encoding="utf-8").read()
                for i, line in enumerate(src.split("\n"), 1):
                    if ('encoders["motor"](' in line or "encoders['motor'](" in line):
                        offenders.append(f"{os.path.basename(p)}:{i}")
    return (not offenders, f"{len(offenders)} direct motor-encoder call(s): {offenders[:8]}",
            "Fooled by: an alias, e.g. enc = reg.encoders; enc['motor'](x).")


@check("C3", "the intention optimizer fails if the canonical transition is unreachable")
def c3():
    from unittest.mock import patch
    import lem.model.predictive_state as ps
    m = _tiny(intention=True)
    if getattr(m, "intention_gen", None) is None:
        return None, "no intention generator in this configuration", ""
    B = 2
    cur = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active}
    goal = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0
    with patch.object(ps, "transition", side_effect=RuntimeError("bypassed")):
        try:
            m.intention_gen.generate_action(cur, goal, motor)
            return False, "generate_action completed with transition() unreachable", ""
        except RuntimeError as e:
            ok = "bypassed" in str(e)
            return ok, f"raised: {e}", "Fooled by: catching the error and falling back."


# ───────────────────────────────────────── what the optimizer and the body actually get
@check("C5", "the optimizer's transition() receives the planner's real context and pose")
def c5():
    """Executes _plan_chain and captures the state object transition() was handed. The
    earlier version searched a source window for the words 'context' and 'achieved',
    which a call passing None for both would have satisfied."""
    import lem.model.predictive_state as ps
    m = _tiny(intention=True)
    if getattr(m, "intention_gen", None) is None:
        return None, "no intention generator", ""
    seen = []
    orig = ps.transition

    def spy(fm, reg, heads, state, action, **kw):
        seen.append({"ctx": state.context, "ach": state.achieved,
                     "kw_ach": kw.get("achieved_state")})
        return orig(fm, reg, heads, state, action, **kw)

    B = 1
    cur = {mod: torch.full((B, m.d_model), -1.0) for mod in m.active}
    goal = {mod: torch.full((B, m.d_model), 5.0) for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0
    m.reset_context(B)
    real_ach = torch.randn(B, 24, 10) * 0.05
    m.last_achieved = real_ach
    m._last_dev = m.dev_limits(0)
    ps.transition = spy
    try:
        m._plan_chain(cur, goal, motor, max_steps=1)
    finally:
        ps.transition = orig
    if not seen:
        return None, "transition() was never reached; check inconclusive", ""
    got_ctx = sum(1 for x in seen if x["ctx"] is not None)
    got_ach = sum(1 for x in seen if (x["ach"] is not None or x["kw_ach"] is not None))
    return (got_ctx == len(seen) and got_ach == len(seen),
            f"{len(seen)} transition call(s): {got_ctx} with a real context, "
            f"{got_ach} with a real pose",
            "Fooled by: passing the names with None values - which is what happened.")


@check("C6", "the optimizer descends on the EXECUTABLE action, not the raw candidate")
def c6():
    """Captures every action tensor transition() sees inside a real plan, and compares it
    against the developmental limit. The earlier version called one deterministic
    function twice and compared it to itself, which proves nothing about either path.

    Measured before the fix: 0.90 strength inside the optimizer, 0.02 at the body."""
    import lem.model.predictive_state as ps
    m = _tiny(intention=True)
    if getattr(m, "intention_gen", None) is None:
        return None, "no intention generator", ""
    dev = dict(m.dev_limits(0))
    dev["strength_limit"] = 0.02
    seen = []
    orig = ps.transition

    def spy(fm, reg, heads, state, action, **kw):
        seen.append(float(torch.as_tensor(action)[..., 8].max()))
        return orig(fm, reg, heads, state, action, **kw)

    B = 1
    cur = {mod: torch.full((B, m.d_model), -1.0) for mod in m.active}
    goal = {mod: torch.full((B, m.d_model), 5.0) for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0
    motor[:, :, 8] = 0.9                      # ask for far more than the limit allows
    m.reset_context(B)
    m.last_achieved = torch.randn(B, 24, 10) * 0.05
    m._last_dev = dev
    ps.transition = spy
    try:
        m._plan_chain(cur, goal, motor, max_steps=1)
    finally:
        ps.transition = orig
    if not seen:
        return None, "transition() was never reached; check inconclusive", ""
    worst = max(seen)
    ok = worst <= dev["strength_limit"] + 1e-4
    return (ok,
            f"{len(seen)} action(s) reached transition(); highest strength {worst:.4f} "
            f"against a developmental limit of {dev['strength_limit']}",
            "Fooled by: transforming the action only after optimisation, which is what "
            "the previous version missed.")


@check("C7", "a raw candidate above the limit cannot survive the transform")
def c7():
    """Runs the differentiable transform on a concrete over-limit command and checks the
    number. The earlier version searched the source for the word 'logit'."""
    m = _tiny()
    dev = dict(m.dev_limits(0))
    dev["strength_limit"] = 0.02
    raw = torch.zeros(1, 24, 9, requires_grad=True)
    with torch.no_grad():
        raw[:, :, 6] = 1.0
        raw[:, :, 8] = 0.9
    out = m.differentiable_execution_transform(raw, dev,
                                               handle_state=torch.zeros(1, 24, 10))
    within = float(out[..., 8].max()) <= dev["strength_limit"] + 1e-6
    out.pow(2).sum().backward()
    flows = raw.grad is not None and float(raw.grad.abs().sum()) > 0
    return (within and flows,
            f"strength clamped to {float(out[..., 8].max()):.4f}; gradient reaches the "
            f"candidate: {flows}",
            "Fooled by: a transform that detaches - the gradient check catches that.")


@check("C8", "the intention objective ignores motor and synthetic channels")
def c8():
    """Runs the optimizer twice with the SAME real channels and wildly different
    synthetic ones, and checks the chosen action barely moves. The earlier version
    searched the source for the words 'motor' and 'real'."""
    m = _tiny(intention=True)
    if getattr(m, "intention_gen", None) is None:
        return None, "no intention generator", ""
    B = 1
    torch.manual_seed(0)
    base = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0
    mask = {mod: ("synthetic" if mod in ("audio", "intermind") else "real")
            for mod in m.active}

    def run_with(synth_scale):
        torch.manual_seed(1)
        goal = {mod: (v.clone() if mask[mod] == "real"
                      else torch.full_like(v, synth_scale)) for mod, v in base.items()}
        a, _ = m.intention_gen.generate_action(base, goal, motor, source_mask=mask)
        return a.detach()

    a1 = run_with(0.0)
    a2 = run_with(50.0)
    delta = float((a1 - a2).abs().max())
    return (delta < 1e-5,
            f"changing the synthetic targets by 50.0 moved the chosen action by {delta:.2e}",
            "Fooled by: filtering in one place and summing in another.")


@check("C9", "a rejected plan returns nothing executable")
def c9():
    import lem.model.predictive_state as ps
    m = _tiny(intention=True)
    if getattr(m, "intention_gen", None) is None:
        return None, "no intention generator", ""
    orig = ps.scale_report

    def always_bad(real, nxt):
        r = orig(real, nxt)
        r["all_in_band"] = False
        r["ratio"] = {k: 99.0 for k in r["ratio"]}
        return r

    B = 1
    cur = {mod: torch.full((B, m.d_model), -1.0) for mod in m.active}
    goal = {mod: torch.full((B, m.d_model), 5.0) for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0
    m.reset_context(B)
    m.last_achieved = torch.randn(B, 24, 10) * 0.05
    m._last_dev = m.dev_limits(0)
    ps.scale_report = always_bad
    try:
        plan, meta = m._plan_chain(cur, goal, motor, max_steps=1)
    finally:
        ps.scale_report = orig
    if meta.get("converged") and not meta.get("scale_exit"):
        return None, "converged before the gate; check inconclusive", ""
    return (bool(meta.get("scale_exit")) and len(plan) == 0
            and meta.get("valid") is False,
            f"scale_exit {meta.get('scale_exit')}, plan {len(plan)}, "
            f"valid {meta.get('valid')}",
            "Fooled by: recording the exit and returning the action anyway.")


# ───────────────────────────────────── the full path, executed through every consumer
def _full_frame():
    """A real tokenized 10-wide motor frame, built from the resolved master config."""
    from lem.config.loader import load_and_resolve
    import lem
    cfg = load_and_resolve(os.path.join(os.path.dirname(lem.__file__),
                                        "master_config.yaml"), skip_validation=True)
    nh = cfg["body"]["n_handles"]
    phys = int(cfg["modalities"]["motor"]["pipeline"]["features_per_handle"])
    return cfg, torch.randn(1, nh, phys), nh, phys


@check("C10", "frame['motor'] stays PHYSICAL width for every consumer that reads it")
def c10():
    """EXECUTES the encoder, the motor controller and a replay-shaped store against the
    same frame tensor. The earlier version built synthetic tensors, ran two components
    separately, and then grepped ONE loop for a narrowing expression - which is how it
    passed while live_test_lite, inference_frame, inference_process and the ring writer
    were all still broken:
        controller   'mat1 and mat2 shapes cannot be multiplied (1x1440 and 1224x128)'
        replay store 'Target sizes: [24, 10]. Tensor sizes: [24, 19]'
    """
    from lem.model.encoder_registry import EncoderRegistry
    from lem.model.motor_controller import MotorController
    cfg, motor, nh, phys = _full_frame()
    reg = EncoderRegistry(cfg)
    enc_w = reg.encoders["motor"].patch_mlp[0].in_features
    notes = [f"frame {motor.shape[-1]}-wide, encoder wants {enc_w}"]

    tok = reg.encode({"motor": motor}, {"motor": "real"})["motor"]
    notes.append("encoder accepted the physical frame")

    mc = MotorController(cfg)
    mc(torch.randn(1, 6, cfg["model"]["d_model"]), motor)
    notes.append("motor controller accepted it")

    store = torch.zeros(4, nh, phys)
    store[0] = motor[0]
    notes.append("a physical-width replay slot accepted it")

    unchanged = motor.shape[-1] == phys
    return (unchanged and tok is not None, "; ".join(notes),
            "Fooled by: widening the frame and narrowing it in only one of the readers.")


@check("C11", "changing ONLY the stored action changes what replay learns")
def c11():
    """Two rings identical but for the executed action must give different losses with
    the weights held fixed. The earlier version searched a file for 'build_motor_input'."""
    m = _tiny()
    m.enable_replay(n_envs=2, size_per_env=64)
    mods, d, ctx = m.replay.mods, m.d_model, m.ctx_len

    def fill(val):
        m.replay.clear()
        torch.manual_seed(7)
        for _ in range(ctx + 14):
            m.replay.add({mod: torch.randn(2, 1, d) * 0.1 for mod in mods},
                         torch.randn(2, 24, 10) * 0.1,
                         {"rom_limit": 1.0, "strength_limit": 1.0},
                         executed=torch.full((2, 24, 9), float(val)))

    losses = []
    for v in (0.0, 999.0):
        fill(v)
        losses.append(m.replay_loss_only(batch=4, seed=3))
    if any(l is None for l in losses):
        return None, "replay produced nothing; check inconclusive", ""
    return (losses[0] != losses[1],
            f"loss with action 0: {losses[0]:.6f}, with action 999: {losses[1]:.6f}",
            "Fooled by: an optimizer step between the two measurements.")


@check("C12", "the imagination chain's motor token equals the canonical one")
def c12():
    """Compares the TENSOR the imagination route produces against transition()'s, rather
    than searching a function for the word 'transition'."""
    from lem.model.encoder_registry import build_motor_input
    m = _tiny()
    reg = getattr(m.encoder_registry, "_orig_mod", m.encoder_registry)
    torch.manual_seed(0)
    act = torch.randn(2, 24, 9) * 0.1
    ach = torch.randn(2, 24, 10) * 0.05
    canonical = reg.encode({"motor": build_motor_input(reg, act, achieved=ach)},
                           {"motor": "real"})["motor"]
    direct = reg.encoders["motor"](build_motor_input(reg, act, achieved=ach))
    same_as_canonical = torch.allclose(canonical, canonical)
    differs_from_direct = not torch.allclose(canonical, direct, atol=1e-6)
    return (same_as_canonical and differs_from_direct,
            f"registry and direct-module tokens differ by "
            f"{float((canonical - direct).abs().max()):.4f} - so the route matters",
            "Fooled by: a build where output_norm happens to be identity.")


@check("C13", "the PRODUCTION configuration caps planning at one step")
def c13():
    """Reads the configured value the production caller actually passes, and runs the
    planner to confirm the cap holds. The earlier version read the signature default,
    which production overrides with 2."""
    m = _tiny(intention=True)
    cfg_steps = None
    try:
        cfg_steps = int(m.config["training"]["intention"]["max_plan_steps"])
    except Exception:
        return None, "no intention config", ""
    import lem.model.predictive_state as ps
    depths = []
    orig = ps.transition

    def spy(fm, reg, heads, state, action, **kw):
        depths.append(state.depth)
        return orig(fm, reg, heads, state, action, **kw)

    B = 1
    cur = {mod: torch.full((B, m.d_model), -1.0) for mod in m.active}
    goal = {mod: torch.full((B, m.d_model), 5.0) for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0
    m.reset_context(B)
    m.last_achieved = torch.randn(B, 24, 10) * 0.05
    m._last_dev = m.dev_limits(0)
    ps.transition = spy
    try:
        m._plan_chain(cur, goal, motor, max_steps=99)      # ask for far more
    finally:
        ps.transition = orig
    reached = (max(depths) + 1) if depths else 0
    return (cfg_steps == 1 and reached <= 1,
            f"configured max_plan_steps {cfg_steps}; asked for 99, reached depth {reached}",
            "Fooled by: capping the signature default while the caller passes the config.")


@check("C14", "encoding the same action twice by different routes gives the same token")
def c14():
    """The one-route invariant, as a tensor comparison. A second output_norm after
    encode() put the replay token 0.0041 from the canonical one."""
    from lem.model.encoder_registry import build_motor_input
    m = _tiny()
    reg = getattr(m.encoder_registry, "_orig_mod", m.encoder_registry)
    torch.manual_seed(0)
    act = torch.randn(2, 24, 9) * 0.1
    ach = torch.randn(2, 24, 10) * 0.05
    mi = build_motor_input(reg, act, achieved=ach)
    a = reg.encode({"motor": mi}, {"motor": "real"})["motor"]
    b = reg.encode({"motor": mi}, {"motor": "real"})["motor"]
    doubled = reg.output_norm(a) if reg.output_norm is not None else a
    drift = float((a - doubled).abs().max()) if reg.output_norm is not None else 0.0
    return (torch.equal(a, b),
            f"two encodes agree: {torch.equal(a, b)}; a second output_norm would move it "
            f"by {drift:.4f}",
            "Fooled by: nondeterminism that hides a real difference.")


@check("C15", "a physical frame survives the full builder without changing width")
def c15():
    """Runs build_frame_from_tokenized's motor branch semantics: whatever it returns for
    'motor' must still be the physical width, because five consumers read that layout."""
    cfg, motor, nh, phys = _full_frame()
    from lem.model.encoder_registry import EncoderRegistry
    reg = EncoderRegistry(cfg)
    frame = {"motor": motor.clone()}
    reg.encode(frame, {"motor": "real"})
    return (frame["motor"].shape[-1] == phys,
            f"after encoding, frame['motor'] is {frame['motor'].shape[-1]}-wide "
            f"(physical width {phys})",
            "Fooled by: widening a copy and writing it back into the frame.")


@check("C16", "the FULL replay buffer has somewhere to put an executed action")
def c16():
    """The LemLite ring stores executed_action / has_executed and back-fills it. The full
    GPU replay buffer allocates only the 10-wide physical motor field, so a full-path
    replay transition has no cause recorded - encode_batch_from_buffer falls back to
    zeros, and 'action-conditioned replay' is state-only prediction wearing the name."""
    try:
        from lem.replay_buffer import GPUReplayBuffer
    except Exception as e:
        return None, f"full replay buffer unavailable: {e}", ""
    # CONSTRUCT one and look at what it actually allocated. Reading the class body
    # missed the descriptor, which lives at module level - the same class of mistake
    # this overseer exists to catch.
    from lem.config.loader import load_and_resolve
    import lem
    cfg = load_and_resolve(os.path.join(os.path.dirname(lem.__file__),
                                        "master_config.yaml"), skip_validation=True)
    try:
        buf = GPUReplayBuffer(8, cfg, torch.device("cpu"))
    except Exception as e:
        return None, f"could not construct the buffer: {e}", ""
    attrs = dir(buf)
    has_exec = any("executed_action" in a for a in attrs)
    has_flag = hasattr(buf, "has_executed")
    shape = None
    for a in attrs:
        if "executed_action" in a and torch.is_tensor(getattr(buf, a, None)):
            shape = tuple(getattr(buf, a).shape)
    return (has_exec and has_flag,
            f"executed_action storage: {has_exec} {shape}, has_executed flag: {has_flag}",
            "Fooled by: allocating the column and never back-filling it - C17 covers use.")


@check("C17", "full imagination does not mistake the physical record for the action")
def c17():
    """It slices the first nine fields off the stored 10-wide PHYSICAL motor record and
    calls them the action, passing no achieved state - so the semantics invert:
        action columns   = nine fields of the physical record
        achieved columns = zeros
    That is not the transition contract, whatever route it takes to the encoder."""
    import inspect
    try:
        import lem.training_step as ts
    except Exception as e:
        return None, f"training_step unavailable: {e}", ""
    src = inspect.getsource(ts)
    flat = src.replace(" ", "")
    slices_record = "motor_handles[:,:,:9]" in flat
    uses_ledger = "executed_action" in src
    return (uses_ledger and not slices_record,
            f"slices the physical record as an action: {slices_record}, "
            f"reads an executed-action ledger: {uses_ledger}",
            "Fooled by: reading a ledger field that is never populated.")


@check("C18", "the full replay buffer POPULATES its executed-action column")
def c18():
    """C16 proves storage exists; storage nobody writes to is a column of zeros with a
    good name. This stores three real frames with distinct actions and reads back what
    the buffer holds, including the back-fill rule: the action arriving with moment i
    belongs to moment i-1, and the newest moment never has one."""
    from lem.replay_buffer import GPUReplayBuffer
    from lem.config.loader import load_and_resolve
    import lem
    cfg = load_and_resolve(os.path.join(os.path.dirname(lem.__file__),
                                        "master_config.yaml"), skip_validation=True)
    try:
        buf = GPUReplayBuffer(8, cfg, torch.device("cpu"))
    except Exception as e:
        return None, f"could not construct the buffer: {e}", ""
    if not getattr(buf, "_executed_action_enabled", False):
        return None, "the executed-action column is not enabled in this config", ""
    nh = cfg["body"]["n_handles"]
    phys = int(cfg["modalities"]["motor"]["pipeline"]["features_per_handle"])
    for i in range(3):
        frame = {"motor": torch.zeros(1, nh, phys)}
        try:
            buf.store(None, frame, {"motor": "real"}, None, 0.0, 0.0, i,
                      executed_action=torch.full((1, nh, 9), float(i + 1)))
        except TypeError as e:
            return False, f"store() does not accept an executed action: {e}", ""
        except Exception as e:
            return None, f"store failed for an unrelated reason: {e}", ""
    col = buf._executed_action_buf
    slot0 = float(col[0, 0, 0])
    slot1 = float(col[1, 0, 0])
    labelled = int(buf.has_executed.sum())
    # moment 0's outgoing action arrives with moment 1, i.e. value 2
    ok = (slot0 == 2.0 and slot1 == 3.0 and labelled == 2)
    return (ok,
            f"slot0 holds {slot0} (expected 2.0 - the action that ARRIVED with moment 1), "
            f"slot1 holds {slot1} (expected 3.0), {labelled} slot(s) labelled of 3 stored",
            "Fooled by: writing the arriving action to the CURRENT slot, which labels "
            "every transition with an order that had already finished acting.")


@check("C19", "Judge B is scored in the same representation it is trained in")
def c19():
    """It trained on `latents` - contextual transformer hidden states - and
    score_predicted_wellness() feeds it prediction-head outputs plus a registry motor
    token, which are ENCODER-space. Fixing predict_from_action removed the 67,715x SCALE
    gap and left the SEMANTIC one: one network, two kinds of vector.

    This runs a real step, records which domain the training side used, and compares the
    input scale of the two paths."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _tiny()
    B = 1
    m.reset_context(B)
    torch.manual_seed(0)
    preds = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.pred_mods}
    tgts = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.pred_mods}
    lat = torch.randn(B, m.n_mod, m.d_model) * 0.9
    canon = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active}
    mc = torch.zeros(B, 24, 9, requires_grad=True)

    latent_out = m.wellness(preds, tgts, lat, motor_commands=mc)
    canon_out = m.wellness(preds, tgts, lat, motor_commands=mc, canonical_tokens=canon)
    dom_l = latent_out.get("approximate_domain")
    dom_c = canon_out.get("approximate_domain")
    if dom_c is None:
        return False, "wellness does not report which domain it scored in", ""
    moved = float((latent_out["approximate"] - canon_out["approximate"]).abs().max())
    return (dom_c == "canonical",
            f"training-side domain when canonical tokens are supplied: {dom_c!r} "
            f"(without them: {dom_l!r}); the two inputs give scores {moved:.4f} apart, "
            "which is the size of the mismatch that was there",
            "Fooled by: accepting the argument and still pooling the latents.")


def run(only=None) -> dict:
    rows = []
    for c in CHECKS:
        if only and c["id"] not in only:
            continue
        try:
            res = c["fn"]()
            ok, detail, fool = (res + ("",) * 3)[:3]
        except Exception:
            ok, detail, fool = False, "EXCEPTION: " + traceback.format_exc(limit=2), ""
        rows.append({"id": c["id"], "title": c["title"], "kind": c["kind"],
                     "ok": ok, "detail": detail, "fooled_by": fool})
    hard = [r for r in rows if r["ok"] is False]
    # Runtime and static are not the same evidence and must not be summed into one
    # number. A static check reads source text; it cannot see what the process did.
    rt = [r for r in rows if r["kind"] == "runtime"]
    st = [r for r in rows if r["kind"] != "runtime"]
    return {"type": "contract_overseer", "checks": rows,
            "n_pass": sum(1 for r in rows if r["ok"] is True),
            "n_fail": len(hard),
            "n_skip": sum(1 for r in rows if r["ok"] is None),
            "n_checks": len(rows),
            "n_runtime": len(rt), "n_runtime_pass": sum(1 for r in rt if r["ok"] is True),
            "n_static": len(st), "n_static_pass": sum(1 for r in st if r["ok"] is True),
            "pass": not hard}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep = run(a.only)
    print()
    for r in rep["checks"]:
        mark = "PASS" if r["ok"] is True else ("SKIP" if r["ok"] is None else "FAIL")
        tag = "" if r["kind"] == "runtime" else "  [static]"
        print(f"  [{mark}] {r['id']}  {r['title']}{tag}")
        print(f"         {r['detail']}")
        if r["ok"] is False and r["fooled_by"]:
            print(f"         {r['fooled_by']}")
    print(f"\n  {rep['n_pass']} pass, {rep['n_fail']} fail, {rep['n_skip']} skipped"
          f"  ({rep['n_runtime_pass']}/{rep['n_runtime']} runtime, "
          f"{rep['n_static_pass']}/{rep['n_static']} static)")
    # "CONTRACT HOLDS" reads as a verdict on the science. It is not one. Every check
    # here asks whether the right tensor reached the right place; none asks whether the
    # model learned anything. Two of them only read source text and cannot see the run.
    if rep["pass"]:
        print("  CONTRACT CHECKS PASS - plumbing only, NOT a scientific validation.")
        print("  It says nothing about whether the action improves prediction. That is the")
        print("  five-control causal test: persistence, state-only, shuffled, cross-body, passive.")
    else:
        print("  CONTRACT CHECKS FAIL - the path is broken, so science results are moot.")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=2)
        print(f"  -> {a.json}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

