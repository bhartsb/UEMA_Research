# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""PredictiveState and transition() — one state representation, used everywhere.

The defect this exists to remove
--------------------------------
The approximate wellness evaluator (Judge B) is TRAINED on contextual hidden states
produced by the full forward transformer, and was USED on something else entirely.
`ForwardModel.predict_from_action()`:

  * called the motor encoder module directly, skipping the registry's output LayerNorm
    that every real token receives;
  * spliced that raw token into `last_step` — tensors that have already been through the
    whole transformer stack — mixing two representation levels in one tensor;
  * ran the prediction heads on that mixture WITHOUT rerunning the temporal and
    cross-modal stack, so no actual state transition happened.

Measured consequence on a live body: the inputs reaching the approximate wellness
evaluator through `score_predicted_wellness` were |mean| 63.2 against 0.048 for the real
path. The scorer railed to exactly 0.0 and 1.0, `motor_state` pinned at 0.0, the 0.05
safety exit fired on 100% of plans, and every plan was cut to length 1. That defect is
real and was measured on a running body.

A CORRECTION TO MY OWN EARLIER NUMBERS. I later reported the legacy route at "67,715x
real encoder tokens" and a "12.4x residual". Both were wrong, and wrong the same way:
the reference was `randn * 0.05` and the states were synthetic. Real encoder tokens go
through the registry's output LayerNorm and measure RMS 1.0000 - measured on a live body,
every modality. Against a REAL state:

    real encoder tokens          RMS 1.0000   1.00x by definition
    legacy predict_from_action   RMS 3.4653   3.5x
    canonical transition         RMS 1.0000   1.00x, every modality in the 0.5-2.0 band

So the legacy route is off by ~3.5x on real input, not by five orders of magnitude, and
the canonical route sits exactly on the real distribution. The "12.4x residual" did not
exist; it was an artefact of comparing a normalised tensor against a fabricated 0.05
baseline. Reporting a ratio without checking what the denominator actually is, is the
same error this module exists to prevent.

A scalar rescale would only have shrunk the numbers while leaving the semantic mismatch
in place, which is why this is a type and a single transition function rather than a
normalisation constant.

The contract
------------
    next_state = transition(state, executed_action, duration_s)

One route. Candidate actions go through the same motor encoder and the same output
LayerNorm as real ones. The full forward model runs. Prediction heads output canonical
normalised modality tokens. Nothing mixes transformer latents with raw encoder tokens.

WHAT THE TWO CALL SITES STILL DO NOT SHARE, AND IT IS NOT SETTLED HERE. The heads'
output is normalised here, before it is graded and before it is fed back; the live
delayed next-frame lesson in mimo_bridge/lem_model.py grades the same heads' output
with no normalisation at all. So the same weights are taught two different functions
of themselves inside one step, and because LayerNorm ignores a per-sample rescale of
its input, the imagination lesson puts no pressure on the SCALE of the heads' raw
output - the next-frame lesson is the only thing holding it. Measured on a fresh build
of the H81 configuration (2026-09-15): normalised against raw head output differs
0.0024-0.0155 RMS per sense against a token RMS of 1.0; the same summed next-frame
lesson reads 8.271588 on the raw output against 8.246622 on the normalised one (0.30%);
and doubling every Linear inside the prediction heads moves the imagination loss by
-0.27 on 24.27 (1.1%) while it moves the next-frame lesson by +1.10 on 8.27 (13%).
Whether the chain should grade the un-normalised output, or the live lesson should
normalise too, is a contract decision and neither has been made.
"""
from __future__ import annotations

import contextlib

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class PredictiveState:
    """A state the forward model can be handed, real or imagined, with no way to tell.

    `tokens` are canonical: encoder outputs after the registry's output LayerNorm, one
    [B, d_model] per active modality. `context` is the temporal buffer as of this state.
    `depth` records how many imagined steps produced it — 0 is real — so a caller can
    report rollout depth without inferring it.
    """

    tokens: Dict[str, torch.Tensor]
    context: Optional[torch.Tensor] = None
    timestamp: float = 0.0
    depth: int = 0
    source_mask: Dict[str, str] = field(default_factory=dict)
    # The achieved handle state this state corresponds to. It lives HERE because
    # transition() needs it to build the motor token, and taking it as a separate
    # argument meant the state did not contain everything required to transition
    # itself: a multistep rollout then reused the SAME real pose from before the
    # rollout at depth 2, 3 and beyond, while the imagined body was supposed to have
    # moved. Carried forward from predicted proprioception where available.
    achieved: Optional[torch.Tensor] = None
    # "real" or "predicted". Copying a real source mask into an imagined state and
    # leaving the domain implicit is how two representations came to be scored as if
    # they were one.
    domain: str = "real"
    memory_bias: Optional[torch.Tensor] = None
    # [B, 6] the head's accelerometer and gyro at this state (the vestibular vector's head fields),
    # read by the efference gaze head under --vestibular-sense on (lem/model/efference.py); None
    # otherwise. Carried unchanged through transition(), stale by `depth` steps like `achieved`.
    head_motion: Optional[torch.Tensor] = None
    # Optional forward output for a physical-goal cost. Populated only when
    # requested, so action search shares the sensory transition's forward pass.
    transition_latent: Optional[torch.Tensor] = None

    @property
    def is_real(self) -> bool:
        return self.domain == "real"

    @property
    def batch(self) -> int:
        for v in self.tokens.values():
            return int(v.shape[0])
        return 0

    def detached(self) -> "PredictiveState":
        return replace(self,
                       tokens={m: v.detach() for m, v in self.tokens.items()},
                       context=None if self.context is None else self.context.detach(),
                       memory_bias=None if self.memory_bias is None else self.memory_bias.detach(),
                       transition_latent=None if self.transition_latent is None else self.transition_latent.detach())

    def rms(self) -> Dict[str, float]:
        """Per-modality RMS. The scale check that would have caught the 1,500x blow-up
        the first time it happened, rather than months later."""
        return {m: float(v.detach().float().pow(2).mean().sqrt())
                for m, v in self.tokens.items()}


def canonical_state(encoder_registry, frame: Dict[str, Any], source_mask=None,
                    context=None, timestamp: float = 0.0, depth: int = 0) -> PredictiveState:
    """Encode a raw frame the way the live step does — through the registry, so the
    output LayerNorm is applied — and wrap it."""
    reg = getattr(encoder_registry, "_orig_mod", encoder_registry)
    enc = reg.encode(frame, source_mask or {})
    return PredictiveState(
        tokens={m: (v.squeeze(1) if v.dim() == 3 else v) for m, v in enc.items()},
        context=context, timestamp=timestamp, depth=depth,
        source_mask=dict(source_mask or {}))


def transition(forward_model, encoder_registry, prediction_heads,
               state: PredictiveState, executed_action: torch.Tensor,
               duration_s: float = 0.05,
               achieved_state: Optional[torch.Tensor] = None,
               advance_context: bool = True, include_latent: bool = False) -> PredictiveState:
    """state + executed_action -> next state. The ONLY transition.

    Used identically by live prediction, the imagination chain, the causal test and
    Judge B, so a state scored in one is the same kind of thing as a state scored in
    another. `executed_action` is not detached: the wellness gradient path runs through
    it.

    UNITS. `executed_action` is in ENCODER units: positions already divided by the
    tokenizer's 200 cm (normalize_executed_command / command_for_encoder), rotation a
    unit quaternion, radius and strength in [0, 1]. A caller holding an execution-form
    command (cm) must normalise at its own boundary - the planner's Adam loop passed cm
    here until 2026-08-31 and descended on a wrongly-encoded imagined route.
    """
    from lem.model.encoder_registry import build_motor_input, imagined_motor_token

    fm = getattr(forward_model, "_orig_mod", forward_model)
    reg = getattr(encoder_registry, "_orig_mod", encoder_registry)
    heads = getattr(prediction_heads, "_orig_mod", prediction_heads)

    # Through the REGISTRY, so the motor token gets the same output LayerNorm as every
    # real token. Calling reg.encoders["motor"] directly is the original defect.
    # the state's own achieved pose, unless the caller overrides with a fresher one
    achieved_state = achieved_state if achieved_state is not None else state.achieved
    motor_input = build_motor_input(reg, executed_action, achieved=achieved_state)
    # an imagined row: with --motor-asked-change on the path beside the encoder runs with its weights detached
    z_motor = imagined_motor_token(reg, motor_input)
    if z_motor.dim() == 3:
        z_motor = z_motor.squeeze(1)

    tokens = {}
    for m in fm._active_modalities:
        if m == "motor":
            tokens[m] = z_motor.unsqueeze(1)
        elif m in state.tokens:
            t = state.tokens[m]
            tokens[m] = t.unsqueeze(1) if t.dim() == 2 else t
        else:
            tokens[m] = torch.zeros(state.batch, 1, fm._d_model,
                                    device=z_motor.device, dtype=z_motor.dtype)

    # The FULL stack runs. predict_from_action skipped it and ran the heads on a
    # hand-edited last_step, so no state transition took place at all.
    memory_kw = {} if state.memory_bias is None else {'memory_bias':state.memory_bias}
    _seq, last_step, ctx_update = fm(tokens, state.context, **memory_kw)

    ctx = state.context
    if advance_context and ctx is not None:
        n_mod = fm._n_modalities
        ctx = torch.cat([ctx[:, n_mod:, :], ctx_update], dim=1)

    # The heads produce every predicted modality EXCEPT motor, so the returned state was
    # incomplete under its own declared invariant: it advertised one token per active
    # modality and returned five of six. The motor token is the one we just built - it is
    # the state's own action-conditioned motor representation - so it belongs in the state.
    # The heads end in an unconstrained linear layer with no output normalisation, while
    # what they PREDICT is an encoder token - which by definition has been through the
    # registry's output LayerNorm. Applying the same normalisation makes a predicted
    # token the same kind of object as a real one, which is the whole claim here, and
    # moves the heads toward their own training target rather than away from it: the
    # next-frame loss already grades them against normalised encoder outputs.
    #
    # Measured on ckpt_step_791750 against a REAL state (not synthetic input):
    #     legacy predict_from_action   RMS 3.4653
    #     canonical, heads normalised  RMS 1.0000, every modality inside 0.5-2.0
    # the state this transition starts FROM: with the residual on the heads predict the
    # change away from it. state.tokens are raw encoder tokens, which is what the target is.
    # --heads-all-slot-path on (lem/model/heads_all_slot_path.py, GRADIENT, version 2): an imagined row passes
    # through each head's all-slot path inside imagined(): the path's weights take the row's gradient as the old
    # head's weights do (a lesson graded on this row trains both parts of the sum it grades), and the path's running
    # centre does not move. Without the path (no such attribute on the heads) this is the call it always was.
    _asp = getattr(heads, "_orig_mod", heads).__dict__.get("all_slot_path")
    with (_asp.imagined() if _asp is not None else contextlib.nullcontext()):
        out_tokens = dict(heads(last_step,
                                current={m: v.reshape(v.shape[0], -1) for m, v in state.tokens.items()},
                                who="predictive_state.transition"))
    # THE GAZE COPY (lem/model/efference.py, --efference-heads on): the explicit eye command, with
    # the head's and the eye's orientation, adds its part to vision's prediction here exactly as it
    # does in the live next-frame lesson, so the planner, its Adam loop, the imagination chain and
    # the probes' rollouts predict vision the way the lesson trains it. Absent (nothing added) in
    # every build without the heads.
    from lem.model.efference import gaze_copy_of
    _gc = gaze_copy_of(prediction_heads)
    if _gc is not None:
        out_tokens = _gc(out_tokens, last_step, executed_action, achieved_state)
    _norm = getattr(reg, "output_norm", None)
    if _norm is not None:
        out_tokens = {m: _norm(v) for m, v in out_tokens.items()}
    if "motor" in fm._active_modalities and "motor" not in out_tokens:
        out_tokens["motor"] = z_motor

    # Carry the achieved pose forward. Proprioception is what the achieved pose IS, but
    # the predicted proprioception token is d_model-wide, not the 10-feature handle
    # layout, so it cannot be decoded back here. Holding the last real pose is an
    # approximation and is recorded as one: depth > 0 with a real `achieved` means the
    # pose is stale by `depth` steps.
    return PredictiveState(tokens=out_tokens, context=ctx,
                           timestamp=state.timestamp + duration_s,
                           depth=state.depth + 1,
                           source_mask=dict(state.source_mask),
                           achieved=(achieved_state if achieved_state is not None
                                     else state.achieved),
                           domain="predicted", memory_bias=state.memory_bias,
                           head_motion=state.head_motion,
                           transition_latent=last_step if include_latent else None)


def scale_report(real: PredictiveState, imagined: PredictiveState) -> Dict[str, Any]:
    """Predicted/real RMS ratio per modality, with a pass band.

    The gate the planner should have had. Outside 0.5–2.0 means the two domains have
    drifted apart again and anything the scorer says about an imagined state is not
    comparable to what it learned on.
    """
    r, i = real.rms(), imagined.rms()
    ratios = {m: (i[m] / r[m] if r.get(m) else float("inf"))
              for m in i if m in r and r[m] > 0}
    return {"real_rms": r, "imagined_rms": i, "ratio": ratios,
            "in_band": {m: bool(0.5 <= v <= 2.0) for m, v in ratios.items()},
            "all_in_band": all(0.5 <= v <= 2.0 for v in ratios.values()) if ratios else False}
