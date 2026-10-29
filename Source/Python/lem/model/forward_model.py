# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
ForwardModel — 3-stage attention transformer for multi-modal temporal processing.

Assembles per-modality tokens into a sequence with learned modality and temporal
embeddings, then processes through N blocks of 3-stage attention + FFN.

Token assembly: [n_active_modalities * context_length, d_model]
  - Each modality contributes 1 token per timestep
  - Learned modality embeddings + learned temporal embeddings

Three-Stage Attention Block (repeated n_layers times):
  1. Cross-modal:       per-timestep, attend across n_mod tokens
  2. Causal-temporal:   per-modality, attend across ctx timesteps (causal mask)
  3. Cross-modal bridge: full sequence, attend across all ctx*n_mod tokens
  FFN: per-token feedforward

Each stage individually disableable via config for ablation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from lem.model.attention_cap import CappedMultiheadAttention


class ThreeStageBlock(nn.Module):
    """Single 3-stage attention block + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_expansion: int,
                 n_modalities: int, context_length: int,
                 enable_cross_modal: bool, enable_causal_temporal: bool,
                 enable_cross_modal_bridge: bool,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.context_length = context_length
        self.enable_cross_modal = enable_cross_modal
        self.enable_causal_temporal = enable_causal_temporal
        self.enable_cross_modal_bridge = enable_cross_modal_bridge

        # Stage 1: Cross-modal (per timestep). Built as a CappedMultiheadAttention (ATTENTION_CAP,
        # lem/model/attention_cap.py): the same parameters, and nn.MultiheadAttention's own forward
        # while it carries no bound (every build without --attention-cap on).
        if enable_cross_modal:
            self.cm_norm = nn.LayerNorm(d_model)
            self.cm_attn = CappedMultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )

        # Stage 2: Causal-temporal (per modality)
        if enable_causal_temporal:
            self.ct_norm = nn.LayerNorm(d_model)
            self.ct_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            # Causal mask: registered as buffer
            causal = torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1,
            )
            self.register_buffer("causal_mask", causal)

        # Stage 3: Cross-modal bridge (full sequence)
        if enable_cross_modal_bridge:
            self.cmb_norm = nn.LayerNorm(d_model)
            self.cmb_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )

        # FFN (always on)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                capture_attention: Union[bool, str] = False,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]],
                           Tuple[torch.Tensor, dict]]:
        """
        Args:
            x: [B, seq_length, d_model] where seq_length = n_mod * ctx
            capture_attention: If True, return temporal attention weights
                from Stage 2 (causal-temporal) alongside the output (the
                contract lem/training_step.py consumes - unchanged).
                If "all", return a dict with every stage's weights (the
                explainer's mode; stages 1 and 3 pay the need_weights=True
                non-fused path only in this mode, never in training).

        Layout: tokens are arranged as [t0_m0, t0_m1, ..., t0_mN, t1_m0, ...]
                i.e. modalities interleaved within each timestep.

        Returns:
            If capture_attention is False: x tensor [B, S, D]
            If capture_attention is True: (x, attn_weights) where
                attn_weights is [B*n_mod, ctx, ctx] or None if stage 2 disabled.
            If capture_attention == "all": (x, weights_dict) with
                cross_modal [B*ctx, n_mod, n_mod], temporal [B*n_mod, ctx, ctx],
                bridge [B, S, S] - each None if its stage is disabled.
                Weights are head-averaged (average_attn_weights=True), like the
                existing Stage 2 capture.
        """
        B, S, D = x.shape
        n_mod = self.n_modalities
        ctx = self.context_length
        capture_all = capture_attention == "all"
        ct_weights = None
        cm_weights = None
        cmb_weights = None

        # Stage 1: Cross-modal attention (per timestep)
        if self.enable_cross_modal:
            # Reshape: [B, ctx, n_mod, D]
            x_cm = x.reshape(B, ctx, n_mod, D)
            # Flatten batch+time: [B*ctx, n_mod, D]
            x_cm = x_cm.reshape(B * ctx, n_mod, D)

            residual = x_cm
            x_cm = self.cm_norm(x_cm)
            if capture_all:
                x_cm, cm_weights = self.cm_attn(x_cm, x_cm, x_cm, need_weights=True,
                                                average_attn_weights=True)
            else:
                x_cm, _ = self.cm_attn(x_cm, x_cm, x_cm, need_weights=False)
            x_cm = x_cm + residual

            # Reshape back: [B, S, D]
            x = x_cm.reshape(B, ctx, n_mod, D).reshape(B, S, D)

        # Stage 2: Causal-temporal attention (per modality)
        if self.enable_causal_temporal:
            # Reshape: [B, ctx, n_mod, D] -> [B, n_mod, ctx, D]
            x_ct = x.reshape(B, ctx, n_mod, D).permute(0, 2, 1, 3)
            # Flatten: [B*n_mod, ctx, D]
            x_ct = x_ct.reshape(B * n_mod, ctx, D)

            residual = x_ct
            x_ct = self.ct_norm(x_ct)
            if capture_attention:
                x_ct, ct_weights = self.ct_attn(
                    x_ct, x_ct, x_ct, attn_mask=self.causal_mask,
                    need_weights=True, average_attn_weights=True)
            else:
                x_ct, _ = self.ct_attn(x_ct, x_ct, x_ct, attn_mask=self.causal_mask,
                                      need_weights=False)
            x_ct = x_ct + residual

            # Reshape back: [B, n_mod, ctx, D] -> [B, ctx, n_mod, D] -> [B, S, D]
            x = x_ct.reshape(B, n_mod, ctx, D).permute(0, 2, 1, 3).reshape(B, S, D)

        # Stage 3: Cross-modal bridge (full sequence attention)
        if self.enable_cross_modal_bridge:
            residual = x
            x = self.cmb_norm(x)
            if capture_all:
                x, cmb_weights = self.cmb_attn(x, x, x, need_weights=True,
                                               average_attn_weights=True)
            else:
                x, _ = self.cmb_attn(x, x, x, need_weights=False)
            x = x + residual

        # FFN (always on)
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x) + residual

        if capture_all:
            return x, {"cross_modal": cm_weights, "temporal": ct_weights,
                       "bridge": cmb_weights}
        if capture_attention:
            return x, ct_weights
        return x


def _invalidate_loaded_embedding_cache(module, incompatible_keys):
    """A load can replace embeddings on an already-used or nested module."""
    module.invalidate_embedding_cache()


class ForwardModel(nn.Module):
    """
    Multi-modal temporal forward model.

    Receives per-modality tokens from encoder registry, assembles into
    a temporal sequence, processes through 3-stage attention blocks,
    and produces predictions via heads + latents for motor/wellness.

    Args:
        config: Fully-resolved config dict.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        fwd_cfg = model_cfg["forward_model"]
        d_model = model_cfg["d_model"]
        n_modalities = fwd_cfg["_n_modalities"]
        context_length = model_cfg["context_length"]
        seq_length = fwd_cfg["_seq_length"]
        stages = fwd_cfg["stages"]

        self._d_model = d_model
        self._n_modalities = n_modalities
        self._context_length = context_length
        self._seq_length = seq_length
        self._active_modalities = config["_active_modalities"]

        # Learned embeddings
        self.modality_embedding = nn.Embedding(n_modalities, d_model)
        self.temporal_embedding = nn.Embedding(context_length, d_model)
        # (2026-08-24) A gaze_action side-projection briefly lived here - gaze glued onto
        # the motor token from outside. Replaced the same day: the eyes are handle 23 of
        # the body (lem/model/body_sets.py), so gaze arrives inside the motor modality
        # like every limb, and needs no special pathway. Old checkpoints carrying
        # gaze_action.* keys load fine (unexpected keys are tolerated).

        # 3-stage attention blocks
        self.blocks = nn.ModuleList([
            ThreeStageBlock(
                d_model=d_model,
                n_heads=fwd_cfg["n_heads"],
                ffn_expansion=fwd_cfg["ffn_expansion"],
                n_modalities=n_modalities,
                context_length=context_length,
                enable_cross_modal=stages["cross_modal"],
                enable_causal_temporal=stages["causal_temporal"],
                enable_cross_modal_bridge=stages["cross_modal_bridge"],
            )
            for _ in range(fwd_cfg["n_layers"])
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # EXP-16: gradient checkpointing — trade recomputation for smaller
        # backward kernels, reducing peak SM demand during training.
        # No effect under torch.no_grad() (inference CUDA graphs).
        self._grad_checkpoint = fwd_cfg.get("gradient_checkpointing", False)

        # Pre-allocated index tensors — avoids torch.arange() allocation on
        # every forward call (~0.5ms × 5 calls/step = 2-3ms saved).
        self.register_buffer("_mod_ids",
                             torch.arange(n_modalities), persistent=False)
        self.register_buffer("_temp_ids",
                             torch.arange(context_length), persistent=False)

        # Max batch size for pre-allocated token assembly buffer.
        # Avoids list+append+stack on every forward call (inference only).
        self._max_batch = 64
        self.register_buffer("_token_assembly_buf",
                             torch.zeros(self._max_batch, n_modalities, d_model),
                             persistent=False)

        # Embedding cache version: track when embeddings need recomputation.
        # During inference (no_grad), embeddings are frozen and the cache is valid.
        # During training, _refresh_embedding_cache() recomputes from live weights
        # on the first forward call after any optimizer step changes them.
        self._emb_cache_valid = False
        self.register_buffer("_mod_emb_expanded",
                             torch.zeros(1, n_modalities, d_model), persistent=False)
        self.register_buffer("_temp_emb_expanded",
                             torch.zeros(1, seq_length, d_model), persistent=False)
        self.register_load_state_dict_post_hook(_invalidate_loaded_embedding_cache)

    def _refresh_embedding_cache(self) -> None:
        """Recompute expanded embedding buffers from live nn.Embedding weights.

        Called once per forward when cache is invalid (after optimizer step or
        first call). Costs one embedding lookup + expand + reshape -- same as
        the original per-call code, but only runs when weights actually change.
        """
        with torch.no_grad():
            self._mod_emb_expanded.copy_(
                self.modality_embedding(self._mod_ids).unsqueeze(0))
            _t = self.temporal_embedding(self._temp_ids)  # [ctx, d]
            _t = _t.unsqueeze(1).expand(
                self._context_length, self._n_modalities, self._d_model
            ).reshape(self._seq_length, self._d_model)
            self._temp_emb_expanded.copy_(_t.unsqueeze(0))
        self._emb_cache_valid = True

    def invalidate_embedding_cache(self) -> None:
        """Mark embedding cache as stale. Call after optimizer.step()."""
        self._emb_cache_valid = False

    def forward(self, tokens,
                context_buffer: Optional[torch.Tensor] = None,
                capture_attention: Union[bool, str] = False,
                memory_bias: Optional[torch.Tensor] = None,
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]]:
        """
        Args:
            tokens: Either:
                - Dict[str, Tensor]: {modality_name: [B, 1, d_model]} from encoder registry.
                - Tensor: [B, n_mod, d_model] pre-assembled token tensor (fast path,
                  skips dict lookup + squeeze + stack overhead).
            context_buffer: [B, (ctx-1)*n_mod, d_model] -- previous context frames.
                           If None, assumes single-frame (no temporal context).
            capture_attention: If True, also return per-layer temporal attention
                weights from Stage 2 (causal-temporal) - the existing contract,
                unchanged. If "all", the 4th return value is instead a list of
                per-layer dicts {cross_modal, temporal, bridge} covering every
                attention stage (the explainer's mode; see ThreeStageBlock).

        Returns:
            latents: [B, seq_length, d_model] -- full sequence output for motor/wellness.
            last_step_tokens: [B, n_mod, d_model] -- last timestep for prediction heads.
            context_update: [B, n_mod, d_model] -- current frame tokens (with modality
                           embeddings, pre-block) for caller to shift into context_buffer.
            attention_weights: (only if capture_attention) list of per-layer attention
                weight tensors [B*n_mod, ctx, ctx] from Stage 2, or per-layer dicts
                when capture_attention == "all".
        """
        # Refresh embedding cache when needed:
        # - Training: always recompute (embeddings change each optimizer step).
        #   Cost: one lookup + expand + reshape (~0.1ms), but only inside
        #   torch.enable_grad() calls (training forward, imagination).
        # - Inference (no_grad): use cached buffers (embeddings are frozen).
        if not self._emb_cache_valid or torch.is_grad_enabled():
            self._refresh_embedding_cache()

        if isinstance(tokens, torch.Tensor):
            # Fast path: caller already assembled [B, n_mod, d_model] tensor.
            # No dict lookup, no squeeze, no stack.
            current_frame = tokens
            B = tokens.shape[0]
        else:
            # Dict path: assemble from {mod: [B, 1, d_model]} entries.
            _first = tokens[self._active_modalities[0]]
            B = _first.shape[0]

            if torch.is_grad_enabled():
                current_frame = torch.stack(
                    [tokens[m].squeeze(1) for m in self._active_modalities], dim=1)
            else:
                buf = self._token_assembly_buf[:B]
                for i, mod_name in enumerate(self._active_modalities):
                    buf[:, i].copy_(tokens[mod_name].squeeze(1))
                current_frame = buf

        # Add modality embeddings.
        #
        # These are nn.Embedding weights, they have requires_grad=True and they ARE in the
        # optimizer's parameter list - but until 2026-08-25 the forward pass only ever
        # added _mod_emb_expanded, a buffer copied under torch.no_grad() in
        # _refresh_embedding_cache. A gradient cannot reach a tensor through a no_grad
        # copy, so both embeddings were FROZEN at random initialisation for every run this
        # project has ever done. Measured: .grad stayed None and the weight moved 0.0 over
        # five optimizer steps. The transformer's only way to tell one modality position
        # from another, or one timestep from another, was a fixed random offset.
        #
        # The cache is kept for the inference path, where it is a real saving and where
        # gradients are irrelevant.
        current_frame = current_frame + (
            self.modality_embedding(self._mod_ids).unsqueeze(0)
            if torch.is_grad_enabled() and self.training else self._mod_emb_expanded)

        # Pre-block current frame for context buffer update (returned to caller)
        current_flat = current_frame.reshape(B, self._n_modalities, self._d_model)

        if context_buffer is not None:
            # Prepend context: [B, (ctx-1)*n_mod, d_model] + [B, n_mod, d_model]
            sequence = torch.cat([context_buffer, current_flat], dim=1)
        else:
            # Single frame: pad to full context length
            # [B, n_mod, d_model] -> repeat for context_length timesteps
            sequence = current_frame.unsqueeze(1).expand(
                B, self._context_length, self._n_modalities, self._d_model
            ).reshape(B, self._seq_length, self._d_model)

        # Add temporal embeddings. Same defect, same fix - see the note above.
        if torch.is_grad_enabled() and self.training:
            _t = self.temporal_embedding(self._temp_ids)                     # [ctx, d]
            _t = _t.unsqueeze(1).expand(self._context_length, self._n_modalities,
                                        self._d_model).reshape(self._seq_length,
                                                               self._d_model)
            sequence = sequence + _t.unsqueeze(0)
        else:
            sequence = sequence + self._temp_emb_expanded

        # Process through blocks
        layer_attn_weights = [] if capture_attention else None
        for block in self.blocks:
            if capture_attention == "all":
                sequence, attn_w = block(sequence, capture_attention="all")
                layer_attn_weights.append(
                    {k: (None if v is None else v.detach()) for k, v in attn_w.items()})
            elif capture_attention:
                result = block(sequence, capture_attention=True)
                sequence, attn_w = result
                if attn_w is not None:
                    layer_attn_weights.append(attn_w.detach())
            elif self._grad_checkpoint and torch.is_grad_enabled():
                # EXP-16: checkpoint each block — recompute activations during
                # backward instead of storing them.  Reduces peak SM demand.
                # use_reentrant=False required for torch.compile compatibility.
                sequence = grad_checkpoint(block, sequence, use_reentrant=False)
            else:
                sequence = block(sequence)

        sequence = self.final_norm(sequence)
        if memory_bias is not None:
            if memory_bias.shape != (B,self._n_modalities,self._d_model):
                raise ValueError('persistent memory bias does not match this body batch')
            sequence = torch.cat([sequence[:,:-self._n_modalities],
                                  sequence[:,-self._n_modalities:]+memory_bias.to(sequence.dtype)],dim=1)

        # Extract last timestep tokens for prediction heads
        last_step = sequence[:, -self._n_modalities:, :]  # [B, n_mod, d_model]

        if capture_attention:
            return sequence, last_step, current_flat, layer_attn_weights
        return sequence, last_step, current_flat

    def predict_from_action(
        self,
        last_step: torch.Tensor,
        motor_out: torch.Tensor,
        encoder_registry,
        prediction_heads,
        achieved_state: Optional[torch.Tensor] = None,
        state_tokens: Optional[Dict[str, torch.Tensor]] = None,
        context_buffer: Optional[torch.Tensor] = None,
        memory_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict sensory consequences of a motor action (for FORWARD mode wellness).

        CANONICAL PATH (2026-08-25). Given `state_tokens`, this delegates to
        lem.model.predictive_state.transition(): the motor token goes through the
        REGISTRY (so it gets the output LayerNorm every real token gets) and the FULL
        temporal and cross-modal stack runs before the prediction heads.

        The legacy path below runs only when no state_tokens are supplied. It encodes the
        motor token directly, splices it into `last_step` - tensors that have already been
        through the whole transformer - and runs the heads on that mixture WITHOUT
        rerunning the stack, so no state transition happens. Measured consequence:
        prediction-head outputs reached the wellness scorer at 1,500-3,400x the scale of
        real encoder tokens, the scorer railed to 0.0/1.0, and every plan was cut to
        length 1. Callers should pass state_tokens.

        CRITICAL: motor_out must NOT be detached. The gradient path is:
          motor_out -> motor_encoder -> replace token -> prediction_heads -> predictions

        Args:
            last_step: [B, n_mod, d_model] from forward().
            motor_out: [B, n_handles, 9] from motor controller (WITH grad).
            encoder_registry: EncoderRegistry (for motor encoder access).
            prediction_heads: PredictionHeads module.

        Returns:
            (predictions, z_motor):
              predictions: {mod_name: [B, d_model]} predicted next-frame embeddings.
              z_motor: [B, d_model] encoded motor token (for score_predicted_wellness).
        """
        if state_tokens is not None:
            from lem.model.predictive_state import PredictiveState, transition
            st = PredictiveState(tokens={m: (v.squeeze(1) if v.dim() == 3 else v)
                                         for m, v in state_tokens.items()},
                                 context=context_buffer, memory_bias=memory_bias)
            nxt = transition(self, encoder_registry, prediction_heads, st, motor_out,
                             achieved_state=achieved_state, advance_context=False)
            from lem.model.encoder_registry import build_motor_input, imagined_motor_token
            _reg = getattr(encoder_registry, '_orig_mod', encoder_registry)
            _zm = imagined_motor_token(_reg, build_motor_input(_reg, motor_out, achieved=achieved_state))
            return nxt.tokens, (_zm.squeeze(1) if _zm.dim() == 3 else _zm)

        motor_idx = self._active_modalities.index("motor")
        B = motor_out.shape[0]

        # Pad motor_out [B, 23, 9] with is_active=1.0 -> [B, 23, 10]
        # One layout for every producer of a motor token - see build_motor_input.
        from lem.model.encoder_registry import build_motor_input, imagined_motor_token
        registry = getattr(encoder_registry, '_orig_mod', encoder_registry)
        motor_input = build_motor_input(registry, motor_out, achieved=achieved_state)
        # through the registry: the encoder module alone skips the output
        # LayerNorm every real token gets, which is how imagined tokens came to
        # sit 67,715x off the scale the wellness scorer was trained on
        motor_token = imagined_motor_token(registry, motor_input)
        z_motor = motor_token.squeeze(1) if motor_token.dim() == 3 else motor_token

        # Functional replacement: detach non-motor tokens, keep z_motor live.
        # torch.stack is a proper autograd op — no in-place assignment on detached
        # leaves, no version counter ambiguity, safe under torch.compile.
        tokens = []
        for i in range(last_step.shape[1]):
            if i == motor_idx:
                tokens.append(z_motor)
            else:
                tokens.append(last_step[:, i, :].detach())
        modified_last_step = torch.stack(tokens, dim=1)  # [B, n_mod, d_model]

        # Run prediction heads on modified last_step
        heads = getattr(prediction_heads, '_orig_mod', prediction_heads)
        predictions = heads(modified_last_step)

        return predictions, z_motor

    def imagine_step(
        self,
        state_tokens: Dict[str, torch.Tensor],
        motor_out: torch.Tensor,
        encoder_registry,
        prediction_heads,
        context_buffer: Optional[torch.Tensor] = None,
        achieved_state: Optional[torch.Tensor] = None,
        memory_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        One imagination chain step: predict next state from current state + motor action.

        Like predict_from_action but does NOT detach non-motor tokens.
        Runs the full forward model (all 8 transformer blocks) at each chain step,
        then prediction heads on the transformer output.
        Full gradient flow enables multi-step imagination training (UEMA Spec §6.3).

        Args:
            state_tokens: {mod_name: [B, d_model]} current state in encoder space.
                          May omit motor (it gets replaced with freshly-encoded motor_out).
            motor_out: [B, n_handles, 9] motor action to imagine executing.
            encoder_registry: EncoderRegistry (for motor encoder access).
            prediction_heads: PredictionHeads module.
            context_buffer: [B, (ctx-1)*n_mod, d_model] temporal context from prior
                            frames. Shifted each step: oldest frame dropped, current
                            appended. NOT detached — gradients flow through the chain.

        Returns:
            (predictions, updated_context_buffer):
                predictions: {mod_name: [B, d_model]} predicted next-frame embeddings.
                updated_context_buffer: [B, (ctx-1)*n_mod, d_model] shifted context
                    with current frame appended, or None if context_buffer was None.
        """
        B = motor_out.shape[0]

        from lem.model.encoder_registry import build_motor_input, imagined_motor_token
        registry = getattr(encoder_registry, '_orig_mod', encoder_registry)
        motor_input = build_motor_input(registry, motor_out, achieved=achieved_state)
        # through the registry: the encoder module alone skips the output
        # LayerNorm every real token gets, which is how imagined tokens came to
        # sit 67,715x off the scale the wellness scorer was trained on
        _zm = imagined_motor_token(registry, motor_input)
        z_motor = _zm.squeeze(1) if _zm.dim() == 3 else _zm

        # Build token dict {mod: [B, 1, d_model]} — NO detach (gradient flows through chain)
        tokens_dict = {}
        for mod in self._active_modalities:
            if mod == "motor":
                tokens_dict[mod] = z_motor.unsqueeze(1)
            else:
                tokens_dict[mod] = state_tokens[mod].unsqueeze(1)  # Live gradient!

        # Run full forward model: modality/temporal embeddings + 8 transformer blocks
        _, last_step, ctx_update = self.forward(tokens_dict, context_buffer=context_buffer,
                                               **({} if memory_bias is None else {'memory_bias':memory_bias}))

        # Shift context: drop oldest frame, append current (no detach — gradient flows)
        new_context = None
        if context_buffer is not None:
            n_mod = self._n_modalities
            new_context = torch.cat([context_buffer[:, n_mod:, :], ctx_update], dim=1)

        # Run prediction heads on transformer output
        heads = getattr(prediction_heads, '_orig_mod', prediction_heads)
        return heads(last_step), new_context
