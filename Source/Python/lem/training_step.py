# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Training-step helpers extracted from replay_train.

Encode batches, build context, run imagination, compute diagnostics,
and execute the full training step.
"""
from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F

from lem.model.wellness import WellnessMode

log = logging.getLogger("replay_train")

# Native C++ ops (optional — GIL-free sync, batched context assembly)
_native = None
try:
    from lem.native_ops import load_native_ops
    _native = load_native_ops()
except Exception:
    pass

from lem.native_ops import backward_gil_free


# ═══════════════════════════════════════════════════════════════════════════════
# Per-body-set PE decomposition (pure instrumentation, no gradient)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_domain_pe(
    batch: dict, target_batch: dict,
    n_handles: int = 24, proprio_features: int = 12,
) -> dict[str, float | None] | None:
    """Compute per-body-set prediction error from raw (pre-encoding) data.

    Compares current-frame vs next-frame raw sensor values for proprioception
    [B, 288] and motor [B, 24, 10] independently, then aggregates per body set.

    Returns {BodySet.name: float} with None for Spine (no handles).
    Returns None if required data is missing.
    """
    from lem.model.body_sets import BodySet, BODY_SET_HANDLES

    proprio = batch.get("proprioception")          # [B, 276]
    target_proprio = target_batch.get("proprioception")  # [B, 276]
    motor = batch.get("motor")                     # [B, 23, 10]
    target_motor = target_batch.get("motor")       # [B, 23, 10]

    if proprio is None or target_proprio is None:
        return None

    with torch.no_grad():
        # Reshape proprio to [B, n_handles, features_per_handle]
        B = proprio.shape[0]
        proprio_3d = proprio.reshape(B, n_handles, proprio_features)
        target_proprio_3d = target_proprio.reshape(B, n_handles, proprio_features)

        # Per-handle proprio PE: [n_handles]
        handle_proprio_pe = (proprio_3d - target_proprio_3d).pow(2).mean(dim=(0, 2))

        # Per-handle motor PE: [n_handles] (if available)
        handle_motor_pe = None
        if motor is not None and target_motor is not None:
            handle_motor_pe = (motor - target_motor).pow(2).mean(dim=(0, 2))

        # Aggregate per body set
        result = {}
        for bs in BodySet:
            indices = BODY_SET_HANDLES[bs]
            if not indices:
                result[bs.name] = None  # Spine has no handles
                continue
            idx = list(indices)
            pe_val = handle_proprio_pe[idx].mean().item()
            if handle_motor_pe is not None:
                pe_val = (pe_val + handle_motor_pe[idx].mean().item()) / 2.0
            result[bs.name] = round(pe_val, 8)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: get vision encoder from registry (handles torch.compile wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_raw_registry(encoder_registry):
    """Unwrap torch.compile to access raw EncoderRegistry."""
    return getattr(encoder_registry, '_orig_mod', encoder_registry)


def _get_vision_encoder(encoder_registry):
    """Get raw SwinPatchEncoder from registry."""
    raw = _get_raw_registry(encoder_registry)
    return raw.encoders["vision"] if "vision" in raw.encoders else None


# ═══════════════════════════════════════════════════════════════════════════════
# Batched encoding from replay buffer
# ═══════════════════════════════════════════════════════════════════════════════

def encode_batch_from_buffer(batch, encoder_registry, device, active, d_model):
    """Re-encode a batch of raw inputs gathered from the replay buffer.

    Vision: head_forward() on cached backbone features (pool + proj only).
    Other modalities: full encoder forward pass.
    All in the autograd graph (caller controls torch.no_grad if needed).

    Args:
        batch: Dict from GPUReplayBuffer.gather_batch().
        encoder_registry: EncoderRegistry (possibly compiled).
        device: Training device.
        active: List of active modality names.
        d_model: Model hidden dimension.

    Returns:
        {mod: [B, 1, d_model]} token dict.
    """
    raw = _get_raw_registry(encoder_registry)
    output_norm = raw.output_norm
    tokens = {}
    B = None

    # Vision: head_forward on cached backbone features.
    # _vision_no_grad: pool attention on [B_ctx, 32K, d_internal] would OOM
    # even under no_grad — microbatch to keep peak allocation manageable.
    # Device: vision encoder may be on cuda:1 (inference) while backbone is on
    # cuda:0 (buffer). Move backbone to encoder device, output back to `device`.
    vision_no_grad = batch.get("_vision_no_grad", False)
    if "vision" in active and batch.get("vision_backbone") is not None:
        backbone = batch["vision_backbone"]  # [B, N, d_internal]
        B = backbone.shape[0]
        vision_enc = raw.encoders["vision"]
        _vis_dev = next(vision_enc.parameters()).device
        if backbone.device != _vis_dev:
            backbone = backbone.to(_vis_dev)
        if vision_no_grad:
            _VIS_MICRO = 4  # Process 4 frames at a time to avoid OOM
            with torch.no_grad():
                chunks = []
                for i in range(0, B, _VIS_MICRO):
                    chunk = vision_enc.head_forward(backbone[i:i + _VIS_MICRO])
                    chunks.append(chunk.detach())
                tok = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        elif B > 32:
            # Large batch with gradients: microbatch to avoid OOM on
            # pool attention over [B, 32K, d_internal] activations.
            _VIS_MICRO_GRAD = 16
            chunks = []
            for i in range(0, B, _VIS_MICRO_GRAD):
                chunks.append(vision_enc.head_forward(backbone[i:i + _VIS_MICRO_GRAD]))
            tok = torch.cat(chunks, dim=0)
        else:
            tok = vision_enc.head_forward(backbone)  # [B, 1, d_output]
        if tok.device != device:
            tok = tok.to(device)
        if output_norm is not None:
            tok = output_norm(tok)
        tokens["vision"] = tok

    # Audio
    if "audio" in active and batch.get("audio") is not None:
        audio = batch["audio"]
        tok = raw.encoders["audio"](audio["spectral"], audio["spatial"])
        if B is None:
            B = tok.shape[0]
        if output_norm is not None:
            tok = output_norm(tok)
        tokens["audio"] = tok

    # Proprioception
    if "proprioception" in active and batch.get("proprioception") is not None:
        tok = raw.encoders["proprioception"](batch["proprioception"])
        if B is None:
            B = tok.shape[0]
        if output_norm is not None:
            tok = output_norm(tok)
        tokens["proprioception"] = tok

    # Motor
    if "motor" in active and batch.get("motor") is not None:
        # The buffer stores the tokenizer's 10 achieved-state fields; the encoder now
        # takes 19 (achieved + executed command). Handing the record straight in is a
        # deterministic shape error, and widening the BUFFER would be wrong - the
        # controller and the motor loss both read the 10-wide layout. Widen at the
        # encoder boundary, through the one canonical layout function.
        from lem.model.encoder_registry import build_motor_input
        _m = batch["motor"]
        _want = raw.encoders["motor"].patch_mlp[0].in_features
        if _m.shape[-1] != _want:
            _exec = batch.get("executed_action")
            if _exec is None:
                # no ledger in this buffer yet: zeros mean "outgoing action not known",
                # the same convention the live pass uses
                _exec = torch.zeros(*_m.shape[:-1], _want - _m.shape[-1],
                                    device=_m.device, dtype=_m.dtype)
            # the ledger holds the EXECUTED order, whose positions are in centimetres
            # while the achieved half arrives from the tokenizer divided by
            # position_range_cm. Concatenating them unscaled put 99.6% of the row's
            # input energy into three columns and hid the action from the model for six
            # runs on the bridge (c5bd2790). One rule, one place.
            from lem.model.encoder_registry import normalize_executed_command
            _m = build_motor_input(raw, normalize_executed_command(_exec), achieved=_m)
        tok = raw.encode({"motor": _m}, {"motor": "real"})["motor"]
        if B is None:
            B = tok.shape[0]
        # NO second output_norm: encode() already applied it. Applying it twice put
        # the replay token 0.0041 from the canonical one - far smaller than the old
        # 1,500x defect, and still a live-versus-replay split, which is precisely
        # what one canonical route exists to prevent.

        tokens["motor"] = tok

    # Tactile
    if "tactile" in active and batch.get("tactile") is not None:
        tok = raw.encoders["tactile"](batch["tactile"])
        if B is None:
            B = tok.shape[0]
        if output_norm is not None:
            tok = output_norm(tok)
        tokens["tactile"] = tok

    # Voice
    if "voice" in active and batch.get("voice") is not None:
        tok = raw.encoders["voice"](batch["voice"])
        if B is None:
            B = tok.shape[0]
        if output_norm is not None:
            tok = output_norm(tok)
        tokens["voice"] = tok

    # Intermind and zero-fill must match encoder output tensor properties
    # (non-leaf, requires_grad=True, correct dtype) for Dynamo guard consistency.
    # Warmup creates all tokens as `randn + _enc_param` (non-leaf, grad=True, bf16).
    # Plain torch.zeros() is leaf/grad=False/float32 — three guard mismatches.
    _fill_dtype = next(iter(tokens.values())).dtype if tokens else torch.float32
    _fill_grad = torch.zeros(1, device=device, dtype=_fill_dtype, requires_grad=True)
    if "intermind" in active and "intermind" not in tokens:
        if B is None:
            B = 1
        tokens["intermind"] = (torch.zeros(B, 1, d_model, device=device, dtype=_fill_dtype)
                               + _fill_grad)

    # Zero-fill any missing active modalities
    if B is not None:
        zero = torch.zeros(B, 1, d_model, device=device, dtype=_fill_dtype) + _fill_grad
        for mod in active:
            if mod not in tokens:
                tokens[mod] = zero

    # Reorder to match `active` — Dynamo pytree guards include dict key order.
    # Warmup uses {mod: ... for mod in active}; mismatched order causes retrace.
    return {mod: tokens[mod] for mod in active if mod in tokens}


# ═══════════════════════════════════════════════════════════════════════════════
# Batched context reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def build_batched_context(context_windows, replay_buffer, encoder_registry,
                          device, active, n_mod, d_model):
    """Build [B, (ctx-1)*n_mod, d_model] context buffer for batched training.

    Context frames are encoded WITH gradients for non-vision modalities
    (encoder learns representations useful for temporal attention).
    Vision uses cached tokens from inference (no_grad, no compute).

    Deduplicates overlapping context windows: 224 total frames often have
    only ~38 unique frames (5.9x reduction in encoder compute).
    """
    ctx_count = len(context_windows[0]) if context_windows else 0
    B = len(context_windows)

    if ctx_count == 0:
        return torch.zeros(B, 0, d_model, device=device)

    # Flatten all context indices
    all_ctx_indices = []
    for ctx_win in context_windows:
        all_ctx_indices.extend(ctx_win)
    total = len(all_ctx_indices)

    use_cache = replay_buffer.has_cached_tokens
    log.info("build_context: has_cached_tokens=%s, token_valid_count=%d, total=%d",
             use_cache, replay_buffer._token_valid.sum().item(), total)

    if use_cache:
        # ── Deduplicated + cached path ──
        # Encode each unique frame only once (6x reduction for overlapping windows)
        _t_ctx0 = time.perf_counter()

        unique_indices = sorted(set(all_ctx_indices))
        idx_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}
        n_unique = len(unique_indices)
        positions = torch.tensor(
            [idx_to_pos[idx] for idx in all_ctx_indices],
            dtype=torch.long, device=device)
        # Pre-build GPU index tensor once — avoids repeated CPU→GPU
        # transfer inside gather_cached_tokens.
        unique_idx_tensor = torch.tensor(unique_indices, dtype=torch.long)

        # Vision: read cached tokens (no_grad, zero compute)
        cached_vision = None
        if "vision" in active:
            with torch.no_grad():
                _cv = replay_buffer.gather_cached_tokens(
                    unique_indices, ["vision"],
                    indices_tensor=unique_idx_tensor)
                if "vision" in _cv:
                    cached_vision = _cv["vision"].squeeze(1)  # [n_unique, d_model]
        _t_ctx1 = time.perf_counter()

        # Non-vision: encode unique frames with gradients (stays in Python for autograd)
        non_vision_active = [m for m in active if m != "vision"]
        unique_nonvis = {}  # {mod: [n_unique, d_model]}
        _gather_ms = 0.0
        _encode_ms = 0.0
        if non_vision_active:
            _CTX_MICRO = 64
            chunk_results = {m: [] for m in non_vision_active}
            for start in range(0, n_unique, _CTX_MICRO):
                chunk_idx = unique_indices[start:start + _CTX_MICRO]
                _tg0 = time.perf_counter()
                ctx_batch = replay_buffer.gather_batch(
                    chunk_idx, skip_vision=True)
                _tg1 = time.perf_counter()
                chunk_tokens = encode_batch_from_buffer(
                    ctx_batch, encoder_registry, device, active, d_model)
                _te1 = time.perf_counter()
                _gather_ms += (_tg1 - _tg0) * 1000
                _encode_ms += (_te1 - _tg1) * 1000
                for m in non_vision_active:
                    if m in chunk_tokens:
                        chunk_results[m].append(chunk_tokens[m].squeeze(1))
                del ctx_batch, chunk_tokens
            for m in non_vision_active:
                if chunk_results[m]:
                    unique_nonvis[m] = torch.cat(chunk_results[m], dim=0)
        _t_ctx2 = time.perf_counter()

        log.info("build_context hot: dedup=%d/%d, vision_cache=%.1fms, "
                 "nonvis_encode=%.1fms (gather=%.1f encode=%.1f), native=%s",
                 n_unique, total,
                 (_t_ctx1 - _t_ctx0) * 1000,
                 (_t_ctx2 - _t_ctx1) * 1000,
                 _gather_ms, _encode_ms,
                 _native is not None)

        # Phase 3b: reassemble (C++ if available, else Python)
        if _native is not None:
            mod_tokens = []
            for mod in active:
                if mod == "vision" and cached_vision is not None:
                    mod_tokens.append(cached_vision)
                elif mod in unique_nonvis:
                    mod_tokens.append(unique_nonvis[mod])
                else:
                    mod_tokens.append(
                        torch.zeros(n_unique, d_model, device=device))
            return _native.assemble_context_tokens(
                mod_tokens, positions, B, ctx_count, n_mod, d_model)
        else:
            # Use index_select (not Python advanced indexing) to create
            # independent tensors.  tensor[positions] creates views that
            # keep the entire parent encoder activation tensor alive in
            # the autograd graph, bloating backward memory by ~40 GiB.
            # The native C++ path uses torch::index_select which avoids this.
            frame_tokens = []
            for mod in active:
                if mod == "vision" and cached_vision is not None:
                    frame_tokens.append(
                        torch.index_select(cached_vision, 0, positions))
                elif mod in unique_nonvis:
                    frame_tokens.append(
                        torch.index_select(unique_nonvis[mod], 0, positions))
                else:
                    frame_tokens.append(
                        torch.zeros(total, d_model, device=device))

            # Stack modalities: [total, n_mod, d_model]
            ctx_assembled = torch.stack(frame_tokens, dim=1)
    else:
        # ── Cold cache fallback: original microbatched encode ──
        _CTX_MICRO = 32
        all_tok_chunks = []
        for start in range(0, total, _CTX_MICRO):
            chunk_indices = all_ctx_indices[start:start + _CTX_MICRO]
            ctx_batch = replay_buffer.gather_batch(chunk_indices)
            ctx_batch["_vision_no_grad"] = True
            chunk_tokens = encode_batch_from_buffer(
                ctx_batch, encoder_registry, device, active, d_model)
            chunk_frame = torch.stack(
                [chunk_tokens[mod].squeeze(1) for mod in active], dim=1)
            all_tok_chunks.append(chunk_frame)
            del ctx_batch, chunk_tokens
        ctx_assembled = torch.cat(all_tok_chunks, dim=0)

    # Reshape to [B, ctx_count * n_mod, d_model]
    ctx_assembled = ctx_assembled.reshape(B, ctx_count, n_mod, d_model)
    ctx_assembled = ctx_assembled.reshape(B, ctx_count * n_mod, d_model)

    return ctx_assembled


# ═══════════════════════════════════════════════════════════════════════════════
# Batched imagination chain
# ═══════════════════════════════════════════════════════════════════════════════

def run_batched_imagination(
    primary_indices, replay_buffer, imagination_steps,
    forward_model, prediction_heads, encoder_registry,
    context_buffer, active, n_mod, d_model, device,
    latent_regularizer=None, latent_reg_weight=0.01,
):
    """Run imagination chain on batched samples from the replay buffer.

    For each sample at buffer index i, imagination targets are at i+1, ..., i+imag_steps.
    Uses compiled forward_model + prediction_heads for speed.

    Pre-batches all motor and target gathers to reduce Python/allocator overhead.

    Returns scalar imagination_loss with live gradients.
    """
    B = len(primary_indices)
    cap = replay_buffer._capacity
    raw = _get_raw_registry(encoder_registry)

    # ── Pre-gather all motor and target data for all steps at once ──
    # Motor: steps 0..S-1 → indices at primary+0, primary+1, ..., primary+(S-1)
    if _native is not None:
        _pri_tensor = torch.tensor(primary_indices, dtype=torch.long)
        _motor_t, _target_t = _native.imagination_index_plan(
            _pri_tensor, imagination_steps, cap)
        # CPU int64 tensors from imagination_index_plan — no CUDA sync
        all_motor_indices = _motor_t.reshape(-1).tolist()
        all_target_indices = _target_t.reshape(-1).tolist()
    else:
        all_motor_indices = []
        all_target_indices = []
        for s in range(imagination_steps):
            all_motor_indices.extend(
                [(idx + s) % cap for idx in primary_indices])
            all_target_indices.extend(
                [(idx + s + 1) % cap for idx in primary_indices])

    # Single batched gather for all motor data: [S*B, 23, 10]
    all_motor_batch = replay_buffer.gather_batch(all_motor_indices, skip_vision=True)
    all_motor_handles = all_motor_batch["motor"]  # [S*B, 23, 10]
    # THE EXECUTED-ACTION LEDGER, gathered alongside the poses it belongs to. The
    # old code read `locals().get("all_executed_actions")` - a name assigned NOWHERE
    # in the repository - so it was None on every call and every imagination step
    # was conditioned on a zero action, silently, past every linter. A ledger row
    # is only usable where has_executed marks it; unmarked rows fall back to the
    # zero convention the live pass uses for "not known".
    all_executed_actions = all_motor_batch.get("executed_action")
    _has_exec = all_motor_batch.get("has_executed")
    if all_executed_actions is not None and _has_exec is not None:
        all_executed_actions = all_executed_actions * _has_exec.reshape(-1, 1, 1).to(
            all_executed_actions.dtype)
    del all_motor_batch

    # Single batched gather for all target tokens
    _target_idx_tensor = torch.tensor(all_target_indices, dtype=torch.long)
    with torch.no_grad():
        if replay_buffer.has_cached_tokens:
            all_target_cached = replay_buffer.gather_cached_tokens(
                all_target_indices, active,
                indices_tensor=_target_idx_tensor)
        else:
            all_target_cached = None

    # Pre-allocate reusable tensors
    n_handles = all_motor_handles.shape[1]
    _is_active = torch.ones(B, n_handles, 1, device=device,
                            dtype=all_motor_handles.dtype)
    # First-step zeros must match warmup token properties: non-leaf with
    # requires_grad=True (warmup uses randn + _enc_param).  Plain zeros
    # are leaf/requires_grad=False — Dynamo guard mismatch → retrace → deadlock.
    _fz_param = torch.zeros(1, device=device, requires_grad=True)
    _first_step_zeros = {mod: torch.zeros(B, 1, d_model, device=device) + _fz_param
                         for mod in active if mod != "motor"}

    # Double-buffered context (avoids torch.cat allocation per step)
    imag_ctx_a = context_buffer.clone()
    imag_ctx_b = torch.empty_like(imag_ctx_a)

    # ── Tensor-indexed imagination: pre-allocate token buffer ──
    # Eliminates per-step dict creation (tokens={}) and dict iteration.
    # Motor index: which slot in [B, n_mod, d_model] is motor.
    _motor_idx = active.index("motor")
    # Head-to-full mapping: head_modalities[i] -> active index (excluding motor)
    _ph = getattr(prediction_heads, '_orig_mod', prediction_heads)
    _head_mods = _ph.head_modalities
    _head_to_full = [active.index(m) for m in _head_mods]

    # Pre-allocate token buffer [B, n_mod, d_model] -- reused every step.
    # Initialize with first-step zeros (non-leaf, requires_grad=True for Dynamo).
    imag_token_buf = torch.zeros(B, n_mod, d_model, device=device) + _fz_param

    imagined_stacked = None  # [B, n_heads, d_model] from prediction_heads
    imag_losses = []

    for s in range(imagination_steps):
        # THE ACHIEVED STATE AND THE ACTION ARE DIFFERENT TENSORS.
        #
        # This sliced the first nine fields off the stored PHYSICAL motor record and
        # called them the action, passing no achieved state - so the semantics inverted:
        #     action columns   = nine fields of the physical record
        #     achieved columns = zeros
        # Routing that through the registry fixed the normalisation and left the meaning
        # backwards, which is worse than an obvious shape error because it runs.
        #
        # The executed action now comes from its own ledger column. Where the ledger is
        # absent (an older buffer), the command columns are ZERO - the same "not known"
        # convention the live pass uses - rather than silently borrowing the pose.
        # NO FUTURE POSES IN THE CHAIN (native review 2026-09-08 finding 4): the
        # old slice `all_motor_handles[s*B:(s+1)*B]` injected the RECORDED future
        # achieved pose at every step s > 0 - real future state leaking into a
        # rollout whose other slots are predictions, so its error could never
        # certify a closed-loop imagined future. The chain now holds the s=0
        # achieved pose throughout (stale is causally clean; predicted pose would
        # need a pose-space decoder that does not exist). The executed-action
        # sequence stays from the ledger - conditioning on the actions taken is
        # the rollout's contract; knowing their outcomes is not.
        achieved = all_motor_handles[0:B]                            # [B, n_handles, 10]
        _exec_all = all_executed_actions
        if _exec_all is not None:
            motor_out = _exec_all[s * B:(s + 1) * B]                 # [B, n_handles, 9]
        else:
            motor_out = torch.zeros(achieved.shape[0], achieved.shape[1], 9,
                                    device=achieved.device, dtype=achieved.dtype)

        from lem.model.encoder_registry import build_motor_input, normalize_executed_command
        # the ledger stores command CENTIMETRES (replay_train unit contract);
        # ordinary replay already normalizes at this boundary and imagination
        # consumed the same column raw - 200x apart (native review finding 2)
        motor_input = build_motor_input(raw, normalize_executed_command(motor_out),
                                        achieved=achieved)
        # through the registry: the encoder module alone skips the output
        # LayerNorm every real token gets, which is how imagined tokens came to
        # sit 67,715x off the scale the wellness scorer was trained on
        z_motor = raw.encode({"motor": motor_input}, {"motor": "real"})["motor"]
        # NO second output_norm: encode() already applied it. Applying it twice put
        # the replay token 0.0041 from the canonical one - far smaller than the old
        # 1,500x defect, and still a live-versus-replay split, which is precisely
        # what one canonical route exists to prevent.
        z_motor = z_motor.squeeze(1) if z_motor.dim() == 3 else z_motor

        # Build token tensor -- no dict creation, no dict iteration.
        # Motor slot: freshly encoded z_motor.
        # Non-motor slots: from previous step's predictions or first-step zeros.
        if imagined_stacked is not None:
            # Scatter prediction head outputs to correct modality slots
            for hi, fi in enumerate(_head_to_full):
                imag_token_buf[:, fi] = imagined_stacked[:, hi]
        imag_token_buf[:, _motor_idx] = z_motor

        # Forward model -- tensor fast path (no dict assembly inside forward()).
        # Bypass CUDA graph (if graphed) to avoid overwriting static buffers.
        if hasattr(forward_model, '_graphed_fn'):
            _fm = forward_model._orig_mod
        else:
            _fm = forward_model
        # torch.enable_grad ensures Dynamo's grad_mode guard matches warmup.
        with torch.enable_grad():
            _, last_step_i, ctx_update_i = _fm(imag_token_buf, imag_ctx_a)
        last_step_i = last_step_i.clone()
        ctx_update_i = ctx_update_i.clone()

        # Update imagination context (double-buffered, no allocation)
        imag_ctx_b[:, :-n_mod, :].copy_(imag_ctx_a[:, n_mod:, :])
        imag_ctx_b[:, -n_mod:, :].copy_(ctx_update_i)
        imag_ctx_a, imag_ctx_b = imag_ctx_b, imag_ctx_a

        # Prediction heads -- stacked tensor output [B, n_heads, d_model].
        # Cast to float32 for Dynamo guard consistency.
        with torch.enable_grad():
            pred_stacked, predicted = _ph.forward_stacked(last_step_i)
        pred_stacked = pred_stacked.clone().float()
        imagined_stacked = pred_stacked

        # Ground truth from pre-gathered targets -- build stacked tensor
        with torch.no_grad():
            if all_target_cached is not None:
                # Stack target tokens in head_modalities order for vectorized MSE
                target_slices = []
                for mod in _head_mods:
                    if mod in all_target_cached:
                        target_slices.append(
                            all_target_cached[mod][s * B:(s + 1) * B].squeeze(1).detach())
                if target_slices:
                    target_stacked = torch.stack(target_slices, dim=1)  # [B, n_heads, d_model]
                else:
                    target_stacked = None
            else:
                # Cold cache fallback: microbatch gather+encode
                _IMAG_MICRO = 32
                target_indices = all_target_indices[s * B:(s + 1) * B]
                target_chunks = []
                for _start in range(0, B, _IMAG_MICRO):
                    _chunk_idx = target_indices[_start:_start + _IMAG_MICRO]
                    _chunk_batch = replay_buffer.gather_batch(_chunk_idx)
                    _chunk_batch["_vision_no_grad"] = True
                    _chunk_tok = encode_batch_from_buffer(
                        _chunk_batch, encoder_registry, device, active, d_model)
                    target_chunks.append(
                        {mod: _chunk_tok[mod] for mod in active if mod in _chunk_tok})
                    del _chunk_batch, _chunk_tok
                # Stack into tensor for vectorized MSE
                target_slices = []
                for mod in _head_mods:
                    if mod in target_chunks[0]:
                        target_slices.append(
                            torch.cat([c[mod] for c in target_chunks], dim=0).squeeze(1).detach())
                target_stacked = torch.stack(target_slices, dim=1) if target_slices else None
                del target_chunks

        # Vectorized MSE loss -- single kernel instead of per-modality loop
        if target_stacked is not None:
            step_loss = F.mse_loss(pred_stacked, target_stacked)
        else:
            step_loss = torch.tensor(0.0, device=device)
        if latent_regularizer is not None:
            step_loss = step_loss + latent_reg_weight * latent_regularizer.penalty(predicted)
        imag_losses.append(step_loss)

    return sum(imag_losses) if imag_losses else torch.tensor(0.0, device=device)


def _serialize_loss_bd(loss_bd: dict | None) -> dict | None:
    """Convert loss breakdown dict (may contain torch.Tensors) to plain floats.

    GIL is released during each .item() sync so inference can run.
    """
    if loss_bd is None:
        return None
    from lem.native_ops import scalar_item_gil_free
    out = {}
    for key, val in loss_bd.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                out[f"pred_{k2}"] = round(
                    scalar_item_gil_free(v2) if isinstance(v2, torch.Tensor)
                    else float(v2), 6)
        elif isinstance(val, torch.Tensor):
            out[key] = round(scalar_item_gil_free(val), 6)
    return out


def _compute_grad_norms(forward_model, encoder_registry, motor_controller,
                        prediction_heads, wellness, gaze_head, voice_head):
    """Compute per-layer and per-group gradient L2 norms.

    Returns dict mapping group/layer names to L2 norm floats.
    Uses a single .tolist() sync point for all values.
    """
    keys, vals = [], []

    # Per-layer forward model norms
    fm = getattr(forward_model, '_orig_mod', forward_model)
    for i, block in enumerate(fm.blocks):
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        if grads:
            norms = torch.stack([g.detach().norm(2) for g in grads])
            keys.append(f"fwd_layer_{i}")
            vals.append(norms.norm(2))

    # Per-group norms
    groups = [
        ("encoders", encoder_registry),
        ("motor_ctrl", motor_controller),
        ("pred_heads", prediction_heads),
        ("wellness", wellness),
        ("gaze", gaze_head),
        ("voice", voice_head),
    ]
    for name, module in groups:
        if module is None:
            continue
        raw = getattr(module, '_orig_mod', module)
        grads = [p.grad for p in raw.parameters() if p.grad is not None]
        if grads:
            norms = torch.stack([g.detach().norm(2) for g in grads])
            keys.append(name)
            vals.append(norms.norm(2))

    if not vals:
        return {}

    # Single sync point: stack all scalar tensors, call .tolist()
    result_tensor = torch.stack(vals)
    result_list = result_tensor.tolist()
    return {k: round(v, 6) for k, v in zip(keys, result_list)}


def _compute_encoder_activations(tokens):
    """Compute per-encoder activation statistics (std, mean).

    Returns dict mapping modality names to {std, mean} dicts.
    Uses a single .tolist() sync point per stat.
    """
    ea_keys, ea_stds, ea_means = [], [], []
    with torch.no_grad():
        for mod, tok in tokens.items():
            ea_keys.append(mod)
            ea_stds.append(tok.detach().std())
            ea_means.append(tok.detach().mean())
    if not ea_stds:
        return {}
    stds = torch.stack(ea_stds).tolist()
    means = torch.stack(ea_means).tolist()
    return {k: {"std": round(s, 6), "mean": round(m, 6)}
            for k, s, m in zip(ea_keys, stds, means)}


def _compute_temporal_attention(layer_attn_weights, context_length):
    """Summarize per-layer temporal attention weights into per-position mass.

    Args:
        layer_attn_weights: list of [B*n_mod, ctx, ctx] tensors (one per layer).
        context_length: number of context timesteps.

    Returns:
        dict with "per_position" (list of floats, attention mass received by
        each timestep position averaged across layers/batch/modalities) and
        "entropy" (attention entropy, higher = more uniform).
    """
    if not layer_attn_weights:
        return None
    with torch.no_grad():
        # Stack: [n_layers, B*n_mod, ctx, ctx]
        stacked = torch.stack(layer_attn_weights)
        # Average across layers and batch*modalities: [ctx, ctx]
        avg = stacked.mean(dim=(0, 1))
        # Per-position: sum of attention mass received (column sums)
        # avg[i, j] = attention from query position i to key position j
        # Column sum = total attention mass received by position j
        per_pos = avg.sum(dim=0)  # [ctx]
        # Normalize to sum to 1
        per_pos = per_pos / per_pos.sum()
        # Entropy: -sum(p * log(p))
        eps = 1e-10
        entropy = -(per_pos * (per_pos + eps).log()).sum()

        return {
            "per_position": [round(v, 4) for v in per_pos.tolist()],
            "entropy": round(entropy.item(), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Training step
# ═══════════════════════════════════════════════════════════════════════════════

def run_training_step(
    replay_buffer, batch_size, config,
    encoder_registry, forward_model, prediction_heads,
    motor_controller, wellness, gaze_head, voice_head, loss_manager,
    optimizer, scheduler, scaler, body_schema,
    latent_regularizer, curiosity_system,
    imagination_enabled, imagination_steps,
    wellness_mode, dev_limits, step,
    use_amp, dtype, device, active, n_mod, d_model,
    all_params, grad_clip=1.0,
    latent_reg_weight=0.01,
    prioritized=False, priority_beta=1.0,
    op_tracker=None,
    pace_ms=0,
    diagnostics=False,
):
    """Run one training step on a sampled mini-batch from the replay buffer.

    Returns (loss, loss_breakdown, grad_norm, per_sample_mse, primary_indices, mean_pe)
    or None if insufficient data. per_sample_mse and primary_indices are only
    populated when prioritized=True. mean_pe is the scalar mean prediction error.
    """
    ctx_len = config["model"]["context_length"]
    imag_steps = imagination_steps if imagination_enabled else 0

    # Sub-step timing: record wall-clock time at each phase boundary.
    # Zero overhead when not profiling (<1µs per perf_counter call).
    _step_timings = {}
    _prev_op = [None, 0.0]  # [name, start_time]

    def _set_op(name):
        if op_tracker is not None:
            op_tracker._current_op = name
        now = time.perf_counter()
        if _prev_op[0] is not None:
            _step_timings[_prev_op[0]] = (now - _prev_op[1]) * 1000
        _prev_op[0] = name
        _prev_op[1] = now

    # 1. Sample indices with context windows
    _set_op("sample")
    sample_result = replay_buffer.sample_with_context(
        batch_size, ctx_len, imagination_steps=imag_steps,
        prioritized=prioritized, beta=priority_beta)
    if sample_result is None:
        return None
    primary_indices, context_windows, is_weights = sample_result
    actual_batch = len(primary_indices)

    # Target indices (next frame for prediction target)
    target_indices = [replay_buffer.successor_index(idx) for idx in primary_indices]

    # Yield GIL after heavy Python sampling so inference thread can proceed
    time.sleep(0)

    _set_op("zero_grad")
    optimizer.zero_grad()
    # Yield GIL: optimizer.zero_grad iterates all params in Python
    time.sleep(0)

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
        # 2. Gather and encode current frames WITH gradients
        _set_op("gather_primary")
        batch = replay_buffer.gather_batch(primary_indices)
        time.sleep(0)  # Yield GIL after gather (heavy allocator + Python dict work)
        _set_op("encode_batch")
        tokens = encode_batch_from_buffer(
            batch, encoder_registry, device, active, d_model)
        # Free vision backbone — only tokens needed from here on (~6 GB at B=128)
        batch.pop("vision_backbone", None)

        # Encoder activation diagnostics (GPU sync — only on diagnostics steps)
        encoder_activations = None
        if diagnostics:
            encoder_activations = _compute_encoder_activations(tokens)

        # 3. Build context buffer WITH gradients
        _set_op("build_context")
        batched_context = build_batched_context(
            context_windows, replay_buffer, encoder_registry,
            device, active, n_mod, d_model)
        time.sleep(0)  # Yield GIL after context build (multiple gathers + encodes)

        # 4. Forward model — clone outputs out of CUDAGraph buffers so they
        #    survive through prediction_heads, gaze_head, wellness, and loss.
        #
        # Eager-mode forward_model() dispatches hundreds of ops through Python,
        # holding GIL for ~50ms.  Inference can't submit its CUDAGraph replay
        # during that window.  Sleep gives inference a guaranteed GIL window to
        # submit GPU work BEFORE the forward dispatch loads the GPU queue.
        # (The GPU handles both workloads concurrently with zero SM contention —
        # proven by backward phase where gpu_sync=0.1ms with both running.)
        _set_op("forward")
        time.sleep(0.010)
        # Temporal attention capture: more expensive (need_weights=True),
        # run less frequently than grad_norms/encoder_activations.
        _capture_attn = diagnostics and (step % 100 == 0)
        # torch.enable_grad ensures Dynamo's grad_mode guard matches the
        # warmup compilation (grad=True), regardless of thread-local state.
        with torch.enable_grad():
            if _capture_attn:
                fm_raw = getattr(forward_model, '_orig_mod', forward_model)
                fwd_result = fm_raw(tokens, batched_context, capture_attention=True)
                latents, last_step, ctx_update, _layer_attn_weights = fwd_result
            else:
                latents, last_step, ctx_update = forward_model(tokens, batched_context)
                _layer_attn_weights = None
        latents = latents.clone()
        last_step = last_step.clone()
        ctx_update = ctx_update.clone()

        # EXP-2 NaN diagnostic: detect graph corruption early
        if hasattr(forward_model, '_graphed_fn'):
            if torch.isnan(latents).any() or torch.isnan(last_step).any():
                log.error("EXP-2: forward_model graph produced NaN outputs "
                          f"at step {step}. Skipping weight staging.")
                return None  # Signal caller to skip this step

        _set_op("pred_heads")
        # 5. Prediction heads — clone dict values out of CUDAGraph buffers
        #    (used by wellness, loss_manager, regularizer, and imagination).
        # Dynamo guard consistency: warmup compiles inside enable_grad().
        # The training thread has grad_enabled=True by default, but Dynamo's
        # GLOBAL_STATE guard distinguishes thread-default from explicit context.
        with torch.enable_grad():
            predictions = prediction_heads(last_step)
        predictions = {k: v.clone() for k, v in predictions.items()}

        # 6. Encode target frames WITHOUT gradients (prevents encoder collapse)
        #    Token caching: if inference already encoded these frames, read
        #    cached tokens directly (zero encoder compute). Fall back to
        #    full re-encoding if cache is cold (e.g. first few steps).
        _set_op("gather_target")
        with torch.no_grad():
            # Always gather non-vision data (motor target needed at step 8)
            target_batch = replay_buffer.gather_batch(
                target_indices, skip_vision=True)
            time.sleep(0)  # Yield GIL after target gather

            _set_op("encode_target")
            if replay_buffer.has_cached_tokens:
                # Fast path: read pre-computed tokens from cache
                _target_idx_t = torch.tensor(target_indices, dtype=torch.long)
                target_tokens = replay_buffer.gather_cached_tokens(
                    target_indices, active,
                    indices_tensor=_target_idx_t)
            else:
                # Cold cache fallback: re-gather with vision + encode
                target_batch_full = replay_buffer.gather_batch(target_indices)
                target_batch_full["_vision_no_grad"] = True
                target_tokens = encode_batch_from_buffer(
                    target_batch_full, encoder_registry, device, active, d_model)
                del target_batch_full
        targets = {mod: target_tokens[mod].squeeze(1).detach()
                   for mod in config["_prediction_head_modalities"]
                   if mod in target_tokens}
        from lem.source_mask import training_source_masks
        batch_source_mask, target_source_mask = training_source_masks(
            batch, target_batch, active)

        _set_op("wellness")
        # 7. Handle state and wellness gradient
        handle_state = batch["motor"]  # [B, 23, 10]

        if wellness_mode != WellnessMode.BASELINE and handle_state is not None:
            wellness.set_developmental_limits(dev_limits["rom_limit"])
            motor_commands = handle_state[:, :, :9].detach().clone().to(dtype).requires_grad_(True)
            bs_coherence = body_schema.coherence_score()
            wellness_out = wellness(
                predictions, targets, latents,
                # native review 2026-09-08 finding 3: without canonical_tokens the
                # approximate evaluator trains on transformer hidden states while
                # score_predicted_wellness scores encoder-space tokens - trained
                # and scored representations must be the same (the bridge already
                # passes these; the native caller did not)
                canonical_tokens={m: (v.squeeze(1) if v.dim() == 3 else v).detach()
                                  for m, v in (tokens or {}).items()},
                capability_features=batch.get("motor_capability"),
                motor_commands=motor_commands,
                body_schema_coherence=bs_coherence,
                raw_tactile=batch.get("tactile"))
            total_wellness = torch.cat(
                list(wellness_out["grounded"].values()), dim=-1).mean()
            # Second-order gradients through wellness → motor.
            # When second_order=True (default): create_graph=True, wellness_grad
            # has requires_grad=True.  The JIT warmup in phase_rebuild.py must
            # match this exactly or Dynamo guards will recompile.
            # When second_order=False: create_graph=False, wellness_grad is
            # detached.  Cuts backward graph ~50%, trading second-order motor
            # signal for lower SM pressure (EXP-15).
            _second_order = config.get("training", {}).get("wellness", {}).get(
                "second_order", True)
            wellness_grad = torch.autograd.grad(
                total_wellness, motor_commands,
                create_graph=_second_order, retain_graph=True)[0]
            if not _second_order:
                wellness_grad = wellness_grad.detach()
        else:
            motor_commands = None
            wellness_out = None
            total_wellness = None
            wellness_grad = None

        _set_op("motor")
        # 8. Motor controller
        if handle_state is not None:
            # Dynamo guard consistency: warmup compiles inside enable_grad().
            with torch.enable_grad():
                motor_out, motor_confidence, motor_meta = motor_controller(
                    latents, handle_state, wellness_grad=wellness_grad)
        else:
            motor_out = None
            motor_confidence = None
            motor_meta = {}

        # Motor target from next frame
        if target_batch.get("motor") is not None:
            motor_target = target_batch["motor"][:, :, :9]
        else:
            motor_target = None

        # 9. Forward wellness (predict sensory consequences of action)
        forward_wellness_loss = None
        if wellness_mode == WellnessMode.FORWARD and motor_out is not None:
            fm_raw = getattr(forward_model, '_orig_mod', forward_model)
            if hasattr(fm_raw, 'predict_from_action'):
                # native review 2026-09-08 finding 1, both halves: the old call
                # passed `context_buffer=context_buffer` where no such local
                # exists (NameError on the first FORWARD step - the real batch
                # context lives in `batched_context`), and it omitted the
                # achieved state that sits right here in handle_state, so the
                # motor encoder filled achieved columns with zeros
                predicted_tokens, z_motor = fm_raw.predict_from_action(
                    last_step, motor_out, encoder_registry, prediction_heads,
                    achieved_state=handle_state,
                    state_tokens={m: v for m, v in (tokens or {}).items()},
                    context_buffer=batched_context)
                w_predicted = wellness.score_predicted_wellness(predicted_tokens, z_motor)
                forward_wellness_loss = -w_predicted.mean()

        _set_op("gaze")
        # 10. Gaze — last_step already cloned above
        gaze_out = None
        gaze_target = None
        if gaze_head is not None:
            # Dynamo guard consistency: warmup compiles inside enable_grad().
            with torch.enable_grad():
                gaze_out = gaze_head(last_step).clone()
            gaze_target = batch["gaze_target"]  # [B, 2]

        _set_op("voice")
        # 10b. Voice -- last_step already cloned above
        # No loss consumes voice_out: LossManager adds the voice term only when a voice_target
        # is supplied, and the call below supplies none. The call stays inside enable_grad()
        # ONLY because phase_rebuild._jit_warmup_training_models compiled it that way (it adds
        # voice_out_warmup.sum() to the warm-up loss). Change both sites together or Dynamo
        # retraces on this thread; test_voice_target_and_motor_parts.py pins the pair.
        voice_out = None
        if voice_head is not None:
            with torch.enable_grad():
                voice_out = voice_head(last_step).clone()

        _set_op("imagination")
        # 11. Imagination (batched) — gated by every_n_steps (EXP-17).
        # live_test.py already had this gating; async path did not.
        imagination_loss = None
        _imag_every_n = config.get("training", {}).get(
            "imagination", {}).get("every_n_steps", 1)
        if (imagination_enabled and imag_steps > 0
                and step % _imag_every_n == 0):
            imagination_loss = run_batched_imagination(
                primary_indices, replay_buffer, imagination_steps,
                forward_model, prediction_heads, encoder_registry,
                batched_context, active, n_mod, d_model, device,
                latent_regularizer=latent_regularizer,
                latent_reg_weight=latent_reg_weight)

        # 12. Mean PE (always computed — used by curiosity and/or curriculum)
        #     Single .item() instead of N per-modality .item() calls (EXP-7)
        #     GIL released during sync so inference can run.
        #     Synthetic (placeholder) modalities are excluded: their targets are noise, so
        #     their PE is always high and would pin the curiosity drive at "overwhelmed".
        with torch.no_grad():
            _synthetic = {mod for sm in target_batch.get("source_masks", []) for mod, v in (sm or {}).items()
                          if v == "synthetic" or v == 2}
            pe_tensors = [F.mse_loss(predictions[m], targets[m])
                          for m in predictions if m in targets and m not in _synthetic]
            if pe_tensors:
                from lem.native_ops import scalar_item_gil_free
                mean_pe = scalar_item_gil_free(
                    torch.stack(pe_tensors).mean())
            else:
                mean_pe = 0.0

        # 12b. Per-body-set PE decomposition (pure instrumentation, no gradient)
        domain_pe = compute_domain_pe(batch, target_batch)

        curiosity_drive = 1.0
        if curiosity_system is not None:
            curiosity_drive = curiosity_system.compute_drive(mean_pe)

        # 13. Latent regularizer
        latent_reg_penalty = None
        if latent_regularizer is not None:
            latent_regularizer.update({mod: tokens[mod].squeeze(1).detach()
                                       for mod in tokens})
            latent_reg_penalty = latent_regularizer.penalty(predictions)

        # 14. Scale wellness by curiosity
        scaled_total_wellness = (
            curiosity_drive * total_wellness
            if wellness_mode == WellnessMode.DIRECT and total_wellness is not None
            else None
        )
        # Forward wellness loss is NOT gated by curiosity. The forward path
        # routes through the trained approximate evaluator (not raw formulas),
        # so the "bored zone" suppression that protects against noisy formula
        # gradients in direct mode is inappropriate here. Without this fix,
        # curiosity_drive=0.0 (when PE < pe_min=0.1) zeroes out the forward
        # wellness loss, killing the motor controller gradient entirely.
        scaled_forward = forward_wellness_loss

        _set_op("loss")
        # 16. Loss computation
        loss, loss_bd = loss_manager.compute(
            predictions, targets, batch_source_mask,
            prediction_source_mask=target_source_mask,
            motor_out=motor_out, motor_target=motor_target,
            wellness=wellness_out,
            developmental_limits=dev_limits,
            current_step=step,
            gaze_out=gaze_out, gaze_target=gaze_target,
            voice_out=voice_out,
            return_breakdown=True,
            total_wellness=scaled_total_wellness,
            forward_wellness_loss=scaled_forward,
            imagination_loss=imagination_loss,
            latent_reg_loss=latent_reg_penalty,
            encoder_outputs=tokens)

        # 16b. Per-sample prediction MSE for priority updates (detached, no grad)
        per_sample_mse = None
        if prioritized:
            with torch.no_grad():
                mse_vals = []
                for mod in predictions:
                    if mod in targets:
                        # [B, d] -> per-sample MSE over feature dim
                        mse_vals.append(
                            F.mse_loss(predictions[mod], targets[mod],
                                       reduction='none').mean(dim=-1))
                if mse_vals:
                    per_sample_mse = torch.stack(mse_vals, dim=0).mean(dim=0)  # [B]

        # 16c. Apply importance sampling weight to loss
        if is_weights is not None:
            loss = loss * is_weights.to(device).mean()

    # Extract wellness component scalars before freeing intermediates.
    # Cheap: just .item() on [1,1] tensors already on GPU, no extra compute.
    wellness_snapshot = None
    if wellness_out is not None:
        try:
            wellness_snapshot = {
                name: score.detach().mean().item()
                for name, score in wellness_out["grounded"].items()
            }
            if total_wellness is not None:
                wellness_snapshot["total"] = total_wellness.detach().item()
        except Exception:
            pass  # Don't let logging crash training

    # Free intermediates before backward to reduce peak memory.
    # These Python references keep tensors alive in the caching allocator
    # even though autograd only needs the computation graph itself.
    del tokens, batched_context, predictions, targets, latents, last_step, ctx_update
    del motor_out, motor_confidence, batch
    try:
        del imagination_loss, latent_reg_penalty, gaze_out, wellness_out
    except NameError:
        pass

    # 17. Backward + optimizer
    _set_op("backward")
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        backward_gil_free(loss)

    # Yield GIL after backward (GPU work queued, let inference run)
    time.sleep(0)

    # Per-layer and per-group gradient norms (GPU sync — only on diagnostics steps)
    grad_norms = None
    if diagnostics:
        grad_norms = _compute_grad_norms(
            forward_model, encoder_registry, motor_controller,
            prediction_heads, wellness, gaze_head, voice_head)

    _set_op("clip_grad")
    from lem.native_ops import clip_grad_norm_gil_free
    pre_clip_norm = clip_grad_norm_gil_free(
        all_params, grad_clip, norm_type=2.0, error_if_nonfinite=False)
    time.sleep(0)  # Yield GIL after clip_grad_norm

    _set_op("optimizer_step")
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()

    # Optional pacing: sync training stream + sleep to give inference an
    # uncontested GPU window.  Placed AFTER optimizer_step so the sync only
    # waits for optimizer kernels (~1-2ms), not the full backward queue
    # (~65ms if placed right after backward).
    if pace_ms > 0:
        _set_op("pace")
        torch.cuda.current_stream().synchronize()
        time.sleep(pace_ms / 1000)

    time.sleep(0)  # Yield GIL after optimizer step

    # Temporal attention summary (computed from forward pass, not backward)
    temporal_attention = None
    if _layer_attn_weights is not None:
        ctx_len = config["model"]["context_length"]
        temporal_attention = _compute_temporal_attention(
            _layer_attn_weights, ctx_len)

    # Finalize last op timing
    if _prev_op[0] is not None:
        _step_timings[_prev_op[0]] = (time.perf_counter() - _prev_op[1]) * 1000

    return (loss.detach(), loss_bd, pre_clip_norm, per_sample_mse,
            primary_indices, mean_pe, _step_timings,
            grad_norms, encoder_activations, temporal_attention,
            wellness_snapshot, domain_pe)
