# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Intention Generator — Differentiable Model Inversion (CF-9, UEMA Spec v3.4 §10).

The forward model learns action→consequence. The intention generator INVERTS this:
given a desired future state, find the motor action that would produce it via
gradient descent optimization in action space, using the frozen forward model as
a differentiable simulator.

Three components:
  IntentionGenerator  — §10.2: per-call Adam optimization over action params
  plan_multistep      — §10.3: iterative multi-step planning with safety exits
  CrossSimulationAttention — §10.4: attend across noisy rollouts for consensus
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def effort_penalty(s_bounded: torch.Tensor, strength_reg: float,
                   per_body: bool = False) -> torch.Tensor:
    """Strength regulariser for the intention loss.

    per_body=False (legacy): sum over batch AND handles. The prediction loss it
    is added to is a MEAN, so replicating one body eight times makes the
    relative effort penalty eight times larger despite identical per-body
    experience (external review 2026-09-08, finding 4) - the planner is pushed
    toward weaker actions purely by body count.
    per_body=True (config training.intention.effort_per_body): per-body sum of
    squared strengths averaged over bodies - invariant to duplicating bodies,
    matching the prediction term's averaging. Flag-gated so the change is an
    explicit declared attempt, never a silent behavior shift mid-run.
    """
    if per_body and s_bounded.dim() >= 2:
        return strength_reg * (s_bounded ** 2).sum(dim=tuple(range(1, s_bounded.dim()))).mean()
    return strength_reg * (s_bounded ** 2).sum()


def prospective_wellness_loss(predictions: Dict[str, torch.Tensor],
                              score_fn, weight: float) -> torch.Tensor:
    """The prospective-wellness term (2026-09-09 plan): score the IMAGINED next
    state with the approximate wellness evaluator and pull the action toward
    futures that are predicted to feel better.

    weight * (1 - mean predicted wellness), graph alive through the predicted
    SENSORY tokens only: the motor token is DETACHED before scoring, because it
    is the action itself - leaving it attached hands the optimizer a direct
    lever to flatter the evaluator without changing any outcome (wireheading;
    adversarial pre-review of the plan). The consequence-mediated path
    action -> transition -> predicted senses -> wellness is the only gradient
    route left open.

    score_fn(sensory_tokens_dict, z_motor) -> [B, n] wellness scores in [0,1].
    """
    z_m = predictions.get("motor")
    sens = {m: v for m, v in predictions.items() if m != "motor"}
    if z_m is None:
        ref = next(iter(predictions.values()))
        z_m = torch.zeros_like(ref)
    w = score_fn(sens, z_m.detach())
    return weight * (1.0 - w.mean())


def _dict_to_tokens(
    encoded_dict: Dict[str, torch.Tensor],
    active_modalities: List[str],
    d_model: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build {mod: [B, 1, d_model]} tokens dict from encoder output dict.

    Suitable as input to ForwardModel.forward().
    """
    B = next(iter(encoded_dict.values())).shape[0]
    tokens = {}
    for mod in active_modalities:
        if mod in encoded_dict:
            tokens[mod] = encoded_dict[mod].unsqueeze(1)  # [B, d_model] -> [B, 1, d_model]
        else:
            tokens[mod] = torch.zeros(B, 1, d_model, device=device)
    return tokens


def _dict_mse(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """MSE between two encoder-output dicts over shared keys."""
    losses = []
    for mod in a:
        if mod in b:
            losses.append(F.mse_loss(a[mod], b[mod]))
    if not losses:
        return torch.tensor(0.0, device=next(iter(a.values())).device)
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Helper: temporarily freeze / unfreeze model parameters
# ---------------------------------------------------------------------------

def _set_requires_grad(modules: List[nn.Module], value: bool) -> None:
    """Set requires_grad on all parameters of the given modules."""
    for module in modules:
        raw = getattr(module, '_orig_mod', module)
        for p in raw.parameters():
            p.requires_grad_(value)


def _freeze_params(modules: List[nn.Module]):
    """Disable grads and RETURN the prior flags, so restoration is exact.
    09-09 review counterexample: the old blanket restore-to-True turned a
    deliberately frozen evaluator weight trainable after one planning call."""
    prior = []
    seen = set()
    for module in modules:
        raw = getattr(module, '_orig_mod', module)
        for p in raw.parameters():
            # capture each parameter ONCE (09-09 feasibility review: a module
            # listed twice - or two modules sharing weights - recorded the
            # already-frozen False as its "prior" flag, and restore left a
            # trainable parameter frozen forever)
            if id(p) in seen:
                continue
            seen.add(id(p))
            prior.append((p, p.requires_grad))
            p.requires_grad_(False)
    return prior


def _restore_params(prior) -> None:
    for p, flag in prior:
        p.requires_grad_(flag)


# ---------------------------------------------------------------------------
# §10.2 — IntentionGenerator
# ---------------------------------------------------------------------------

class IntentionGenerator(nn.Module):
    """Differentiable model inversion via per-call Adam optimization.

    NOT a trained neural network — this is gradient descent in action space at
    inference time, using the frozen forward model as a differentiable simulator.
    No trainable parameters of its own: the forward model, encoder registry and
    prediction heads it optimises through are HELD, not registered as children, so
    parameters() and state_dict() stay empty and nothing that walks this module can
    count those 2,856,832 weights a second time.
    """

    def __init__(
        self,
        config: dict,
        forward_model: nn.Module,
        encoder_registry: nn.Module,
        prediction_heads: nn.Module,
    ):
        super().__init__()
        # HELD, NOT REGISTERED. nn.Module.__setattr__ would make these three children,
        # and parameters()/state_dict() would then report weights this module neither
        # owns nor trains - 2,856,832 of them in the H83 checkpoint at step 4,097,248
        # (encoder registry 1,647,616 + forward model 711,936 + prediction heads
        # 497,280). Nothing walks it today, so this changes no number; it removes the
        # double count a future state_dict save, parameter census or shared grad-clip
        # denominator would make.
        object.__setattr__(self, "_forward_model", forward_model)
        object.__setattr__(self, "_encoder_registry", encoder_registry)
        object.__setattr__(self, "_prediction_heads", prediction_heads)

        intention_cfg = config.get("training", {}).get("intention", {})
        self._n_handles = config["body"]["n_handles"]
        self._n_steps = intention_cfg.get("n_steps", 50)
        self._lr = intention_cfg.get("lr", 0.1)
        self._strength_reg = intention_cfg.get("strength_reg", 0.01)
        self._effort_per_body = bool(intention_cfg.get("effort_per_body", False))
        self._d_model = config["model"]["d_model"]
        self._active_modalities = config["_active_modalities"]
        # prospective wellness (2026-09-09 plan, flag-gated, default OFF): the
        # builder injects the score function and freeze list because the
        # evaluator lives outside this module (LemLite wires it)
        self._wellness_score_fn = None       # (sensory_tokens, z_motor) -> [B, n]
        self._wellness_w = float(intention_cfg.get("prospective_wellness_w", 0.0))
        self._wellness_freeze: List[nn.Module] = []

    def generate_action(
        self,
        current_encoded: Dict[str, torch.Tensor],
        desired_encoded: Dict[str, torch.Tensor],
        current_motor: torch.Tensor,
        achieved_state: Optional[torch.Tensor] = None,
        context_buffer: Optional[torch.Tensor] = None,
        source_mask: Optional[Dict[str, str]] = None,
        execution_transform=None,
        dev_limits: Optional[dict] = None,
        memory_bias: Optional[torch.Tensor] = None,
        action_space=None,
        physical_goal=None,
    ) -> Tuple[torch.Tensor, dict]:
        """Optimize a motor action to minimize prediction error toward desired state.

        Args:
            current_encoded: {mod_name: [B, d_model]} current encoder outputs.
            desired_encoded: {mod_name: [B, d_model]} target encoder outputs.
            current_motor: [B, n_handles, 9] current motor state (initialization).
            action_space: None (the default, every existing caller) descends on the 26 x 9
                order directly. Given (lem/model/motor_synergy.SynergyActionSpace, --motor-synergy
                on), the descent runs on ITS parameters - a compact intention z, the strength
                logits and the eyes' orientation - and the order is what its assemble() builds
                from them; everything after the order (execution transform, smoothing,
                transition, goal distance, effort) is the same.

        Returns:
            (action, metadata):
                action: [B, n_handles, 9] optimized motor action (detached).
                metadata: dict with first_loss, last_loss, n_steps.
        """
        device = current_motor.device
        self._source_mask = dict(source_mask or {})
        fm_raw = getattr(self._forward_model, '_orig_mod', self._forward_model)

        # THE CANONICAL STATE. Every Adam iteration below transitions from this, through
        # the one route: registry-encoded motor token, full transformer, prediction heads.
        #
        # Until 2026-08-25 this loop called predict_from_action() without state tokens -
        # the legacy fallback, which encodes the motor token past the registry LayerNorm
        # and runs the heads on a hand-edited last_step without rerunning the stack. So
        # the action was CHOSEN on the broken route and only afterwards scored on the good
        # one. A reviewer proved it by replacing transition() with a function that raises:
        # generate_action still completed. The gradient the canonical route offers was
        # not the gradient the optimizer was descending.
        from lem.model.predictive_state import PredictiveState, transition
        from lem.model.encoder_registry import normalize_executed_command
        state = PredictiveState(
            tokens={m: (v.squeeze(1) if v.dim() == 3 else v)
                    for m, v in current_encoded.items()},
            context=context_buffer, memory_bias=memory_bias)

        # Initialize 4 parameter groups from current_motor
        pos = current_motor[:, :, :3].detach().clone().requires_grad_(True)
        rot = current_motor[:, :, 3:7].detach().clone().requires_grad_(True)
        # LOGIT, not the raw value. The motor controller already emits radius and
        # strength through a sigmoid, so they arrive in [0,1] - and this loop applied
        # sigmoid AGAIN. A strength of 0.02 became sigmoid(0.02) = 0.505, so the planner
        # imagined half strength while the developmental path clamped the body to 0.02.
        # A 25x difference between the action scored and the action performed.
        def _logit(x):
            return torch.log(x.clamp(1e-4, 1 - 1e-4) / (1 - x.clamp(1e-4, 1 - 1e-4)))

        radius = _logit(current_motor[:, :, 7:8].detach()).clone().requires_grad_(True)
        strength = _logit(current_motor[:, :, 8:9].detach()).clone().requires_grad_(True)

        # The eyes are handle 23 of the body, so gaze descends inside `rot` with every
        # other joint - no separate machinery. (A separate tanh-bounded gaze variable
        # lived here for a few hours on 2026-08-24; retired the same day.)
        if action_space is None:
            action_params = [pos, rot, radius, strength]
        else:
            # the synergy's parameters replace the raw order's; the module that turns them into
            # an order is frozen like the forward model, so planning never trains it
            action_params = action_space.init()
        optimizer = torch.optim.Adam(action_params, lr=self._lr)

        # Freeze model params to avoid gradient storage (graph still flows).
        # The wellness evaluator joins the freeze when the prospective term is
        # wired: its params must never accumulate grads from planning. Prior
        # flags are captured and restored EXACTLY on exit.
        frozen_modules = [self._forward_model, self._encoder_registry,
                          self._prediction_heads] + list(self._wellness_freeze)
        from lem.model.efference import gaze_copy_of
        gaze_copy = gaze_copy_of(self._prediction_heads)
        if gaze_copy is not None:
            frozen_modules.append(gaze_copy.heads)
        if action_space is not None:
            frozen_modules = frozen_modules + list(action_space.frozen_modules())
        if physical_goal is not None:
            frozen_modules += list(physical_goal.frozen_modules())
        _prior_flags = _freeze_params(frozen_modules)

        B = current_motor.shape[0]
        first = best = final_attempt = None
        best_action = None
        best_params = [p.detach().clone() for p in action_params]
        best_step = torch.zeros(B, device=device, dtype=torch.long)
        stopped_by = "step_budget"
        updates = 0
        trace = []
        try:
            # Evaluate the starting command AND every updated command. Selection
            # is per body: a batch average must not conceal another body's loss.
            for _step in range(max(0, self._n_steps) + 1):
                optimizer.zero_grad()

                # Assemble action with activations
                space_pen = None
                if action_space is None:
                    rot_norm = F.normalize(rot, dim=-1)
                    r_bounded = torch.sigmoid(radius)
                    s_bounded = torch.sigmoid(strength)
                    action = torch.cat([pos, rot_norm, r_bounded, s_bounded], dim=-1)
                else:
                    action, s_bounded, space_pen = action_space.assemble()
                raw_action = action

                # THE EXECUTABLE FORM of the candidate, not the raw one. Without this
                # the optimizer descends on consequences of a command the body cannot
                # perform: measured 0.90 strength inside the loop against 0.02 at the
                # body, a 45x difference in the quantity being optimised. The transform
                # must be differentiable or the gradient stops here.
                if execution_transform is not None:
                    action = execution_transform(action, dev_limits or {},
                                                 handle_state=achieved_state)
                    # The transform emits EXECUTION units - positions in centimetres.
                    # The motor encoder's contract is the tokenizer's: positions / 200
                    # (the 2026-08-27 centimetre defect - one rule at every encoder
                    # entry). Without this line the Adam loop CHOSE actions on a
                    # cm-encoded imagined route while _plan_chain SCORED the finished
                    # action through command_for_encoder: descending on one encoding,
                    # measured on another. The division keeps the gradient.
                    action = normalize_executed_command(action)
                # Predict consequences THROUGH THE CANONICAL TRANSITION.
                predicted_state = transition(
                    self._forward_model, self._encoder_registry, self._prediction_heads,
                    state, action, achieved_state=achieved_state,
                    advance_context=False, include_latent=physical_goal is not None)
                predictions = predicted_state.tokens

                # The objective is the desired real sensory CONSEQUENCE - not the
                # action, and not noise.
                #
                # transition() returns the motor token, and the goal ring stores every
                # modality including synthetic audio and intermind. Summing MSE over all
                # shared keys asked the optimizer to reproduce its own action (whose real
                # live counterpart has ZERO command columns, because the outgoing action
                # has not been chosen yet) and to match a fresh Gaussian draw. Motor
                # effort belongs in the explicit regulariser below, not smuggled in as a
                # reconstruction target.
                _obj = {m: v for m, v in predictions.items()
                        if m != "motor" and m in desired_encoded
                        and str(getattr(self._source_mask.get(m, "real"), "value",
                                        self._source_mask.get(m, "real"))) == "real"}
                goal = sum(((v - desired_encoded[m]) ** 2).reshape(B, -1).mean(-1)
                           for m, v in _obj.items()) if _obj else torch.zeros(B, device=device)
                if physical_goal is not None:
                    goal = physical_goal(predicted_state.transition_latent, action, achieved_state, goal)
                effort = self._strength_reg * s_bounded.square().reshape(B, -1).sum(-1)
                # Preserve both existing objective conventions exactly.
                if not self._effort_per_body:
                    effort = effort * B
                extra = torch.zeros_like(effort)
                if space_pen is not None:
                    if not hasattr(action_space, "penalty_per_body"):
                        raise TypeError("Action-space penalties must supply penalty_per_body()")
                    extra = action_space.penalty_per_body()
                wellness = torch.zeros_like(effort)
                if self._wellness_score_fn is not None and self._wellness_w > 0:
                    z_m = predictions.get("motor", torch.zeros_like(next(iter(predictions.values()))))
                    sensory = {m: v for m, v in predictions.items() if m != "motor"}
                    w = self._wellness_score_fn(sensory, z_m.detach())
                    wellness = self._wellness_w * (1.0 - w.reshape(B, -1).mean(-1))
                total = goal + effort + extra + wellness
                parts = torch.stack([total, goal, effort, extra, wellness], dim=-1).detach()
                finite = torch.isfinite(parts).all(-1) & torch.isfinite(raw_action).reshape(B, -1).all(-1)
                if first is None:
                    if not bool(finite.all()):
                        raise RuntimeError("Planner starting command has a non-finite objective")
                    first = parts.clone()
                    best = parts.clone()
                    best_action = raw_action.detach().clone()
                improved = finite & (parts[:, 0] < best[:, 0])
                best = torch.where(improved[:, None], parts, best)
                best_action = torch.where(improved[:, None, None], raw_action.detach(), best_action)
                best_step = torch.where(improved, _step, best_step)
                for saved, param in zip(best_params, action_params):
                    take = improved.reshape(B, *([1] * (param.dim() - 1)))
                    saved.copy_(torch.where(take, param.detach(), saved))
                final_attempt = parts
                if getattr(self, "_record_optimizer_trace", False):
                    trace.append(parts.cpu().tolist())
                if not bool(finite.all()):
                    stopped_by = "nonfinite_candidate"
                    break
                if _step >= self._n_steps:
                    break
                loss = total.mean()
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in action_params):
                    stopped_by = "nonfinite_gradient"
                    break
                optimizer.step()
                updates += 1
            # Restore selected parameters too: synergy summaries must describe
            # the returned action, not an abandoned later attempt.
            with torch.no_grad():
                for param, saved in zip(action_params, best_params):
                    param.copy_(saved)
        finally:
            # CRITICAL: always restore the EXACT prior flags, even on exception
            _restore_params(_prior_flags)

        def values(t):
            return [float(v) if torch.isfinite(v) else None for v in t.cpu()]
        _space = {} if action_space is None else {"synergy": action_space.summary()}
        return best_action, {**_space,
            "goal_kind": "physical_pose_250ms" if physical_goal is not None else "sensory_memory",
            "first_loss": float(first[:, 0].mean()),
            "last_loss": float(best[:, 0].mean()),
            "n_steps": self._n_steps,
            "updates": updates, "stopped_by": stopped_by,
            "selection": "best_evaluated_per_body_v1",
            "selected_iteration": best_step.cpu().tolist(),
            "first_goal_distance": float(first[:, 1].mean()),
            "last_goal_distance": float(best[:, 1].mean()),
            "first_effort": float(first[:, 2].mean()),
            "last_effort": float(best[:, 2].mean()),
            "strength_mean": float(best_action[..., 8].mean()),
            "per_body": {name: {"start": values(first[:, i]), "selected": values(best[:, i]),
                                "final_attempt": values(final_attempt[:, i])}
                         for i, name in enumerate(("objective", "goal", "effort", "action_space", "wellness"))},
            **({"optimizer_trace": trace} if trace else {}),
        }


# ---------------------------------------------------------------------------
# §10.3 — plan_multistep
# ---------------------------------------------------------------------------

def plan_multistep(
    intention_generator: IntentionGenerator,
    forward_model: nn.Module,
    encoder_registry: nn.Module,
    prediction_heads: nn.Module,
    approx_wellness: Callable,
    current_state: Dict[str, torch.Tensor],
    goal_state: Dict[str, torch.Tensor],
    current_motor: torch.Tensor,
    max_steps: int = 5,
) -> Tuple[List[torch.Tensor], dict]:
    """Iteratively plan a sequence of actions toward a goal state.

    Args:
        intention_generator: IntentionGenerator instance.
        forward_model: ForwardModel (for predict_from_action).
        encoder_registry: EncoderRegistry (for motor encoding).
        prediction_heads: PredictionHeads module.
        approx_wellness: callable(predicted_tokens, z_motor) -> [B, 5].
        current_state: {mod_name: [B, d_model]} starting encoder outputs.
        goal_state: {mod_name: [B, d_model]} target encoder outputs.
        current_motor: [B, n_handles, 9] starting motor state.
        max_steps: maximum plan length.

    Returns:
        (plan, metadata):
            plan: list of [B, n_handles, 9] action tensors.
            metadata: dict with plan_length, converged, wellness_exit, final_mse.
    """
    plan: List[torch.Tensor] = []
    imagined = current_state
    motor = current_motor
    converged = False
    wellness_exit = False
    final_mse = None

    fm_raw = getattr(forward_model, '_orig_mod', forward_model)
    d_model = intention_generator._d_model
    active_mods = intention_generator._active_modalities
    device = current_motor.device

    for _step in range(max_steps):
        # Check convergence
        with torch.no_grad():
            mse = _dict_mse(imagined, goal_state).item()
        final_mse = mse
        if mse < 0.01:
            converged = True
            break

        # Generate action (manages its own gradient context internally —
        # needs live gradients for inner Adam, cannot be inside no_grad)
        action, _ = intention_generator.generate_action(
            imagined, goal_state, motor)
        plan.append(action)

        # Advance imagined state: run forward model to get last_step,
        # then predict_from_action to get new predictions
        with torch.no_grad():
            tokens_dict = _dict_to_tokens(imagined, active_mods, d_model, device)
            _, last_step, _ = fm_raw.forward(tokens_dict)
            predictions, z_motor = fm_raw.predict_from_action(
                last_step, action, encoder_registry, prediction_heads,
                state_tokens={m: v for m, v in imagined.items()})

            # Safety check: wellness
            w_scores = approx_wellness(predictions, z_motor)  # [B, 5]
            if w_scores.min().item() < 0.2:
                wellness_exit = True
                break

            # Advance state (detach to avoid graph accumulation across steps)
            imagined = {mod: v.detach() for mod, v in predictions.items()}
            motor = action.detach()

    metadata = {
        "plan_length": len(plan),
        "converged": converged,
        "wellness_exit": wellness_exit,
        "final_mse": final_mse,
    }
    return plan, metadata


# ---------------------------------------------------------------------------
# §10.4 — CrossSimulationAttention
# ---------------------------------------------------------------------------

class CrossSimulationAttention(nn.Module):
    """Attend across multiple noisy rollouts of the same goal to find invariants.

    This is the ONLY part of the intention system with trainable parameters
    (the multi-head attention layer and action projection).
    """

    def __init__(self, config: dict):
        super().__init__()
        intention_cfg = config.get("training", {}).get("intention", {})
        d_model = config["model"]["d_model"]
        n_handles = config["body"]["n_handles"]

        self._n_rollouts = intention_cfg.get("cross_sim_rollouts", 5)
        self._noise = intention_cfg.get("cross_sim_noise", 0.1)
        self._d_model = d_model
        self._n_handles = n_handles

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True)
        self.action_proj = nn.Linear(n_handles * 9, d_model)

    def forward(
        self,
        intention_generator: IntentionGenerator,
        forward_model: nn.Module,
        encoder_registry: nn.Module,
        prediction_heads: nn.Module,
        approx_wellness: Callable,
        current_state: Dict[str, torch.Tensor],
        goal_state: Dict[str, torch.Tensor],
        current_motor: torch.Tensor,
        max_plan_steps: int = 5,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], dict]:
        """Run multiple noisy rollouts and attend across them for consensus.

        Returns:
            (consensus_plan, attention_weights, metadata)
        """
        B = current_motor.shape[0]
        device = current_motor.device

        all_plans: List[List[torch.Tensor]] = []
        plan_metadata: List[dict] = []

        for i in range(self._n_rollouts):
            if i == 0:
                noisy_goal = goal_state
            else:
                noisy_goal = {
                    mod: goal_state[mod] + torch.randn_like(goal_state[mod]) * self._noise
                    for mod in goal_state
                }

            plan_i, meta_i = plan_multistep(
                intention_generator, forward_model, encoder_registry,
                prediction_heads, approx_wellness,
                current_state, noisy_goal, current_motor,
                max_steps=max_plan_steps)
            all_plans.append(plan_i)
            plan_metadata.append(meta_i)

        # Find max plan length across rollouts
        max_len = max((len(p) for p in all_plans), default=0)
        if max_len == 0:
            return [], None, {
                "n_rollouts": self._n_rollouts,
                "plan_lengths": [0] * self._n_rollouts,
                "empty": True,
            }

        # Encode actions: project [n_handles * 9] -> d_model
        encoded_rollouts = []
        for plan_i in all_plans:
            encoded_steps = []
            for action in plan_i:
                flat = action.reshape(B, -1)  # [B, n_handles * 9]
                encoded_steps.append(self.action_proj(flat))  # [B, d_model]
            # Pad shorter plans with zeros
            while len(encoded_steps) < max_len:
                encoded_steps.append(
                    torch.zeros(B, self._d_model, device=device))
            encoded_rollouts.append(
                torch.stack(encoded_steps, dim=1))  # [B, max_len, d_model]

        # Stack: [n_rollouts, B, max_len, d_model]
        stacked = torch.stack(encoded_rollouts, dim=0)

        # Reshape for cross-attention: [B, n_rollouts * max_len, d_model]
        stacked_flat = stacked.permute(1, 0, 2, 3).reshape(
            B, self._n_rollouts * max_len, self._d_model)

        # Query: mean across rollouts [B, max_len, d_model]
        query = stacked.mean(dim=0)

        # Cross-attention
        consensus, attn_weights = self.cross_attn(
            query, stacked_flat, stacked_flat)

        metadata = {
            "n_rollouts": self._n_rollouts,
            "plan_lengths": [m["plan_length"] for m in plan_metadata],
            "convergence": [m.get("converged", False) for m in plan_metadata],
            "wellness_exits": [m.get("wellness_exit", False)
                               for m in plan_metadata],
            "consensus_shape": list(consensus.shape),
        }

        # Return first rollout's plan as the executable plan,
        # plus the attention-weighted consensus for analysis
        return all_plans[0], attn_weights, metadata
