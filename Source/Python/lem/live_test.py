# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Live test — run LEM training against UE5 pipeline.

Requires UE to be running with the pipeline producing frames via SHM.

Usage:
    python -m lem.live_test [--steps N] [--device cuda:0] [--no-vision]
    python -m lem.live_test --steps 100 --save-every 50
    python -m lem.live_test --resume checkpoints/ckpt_step_100 --steps 100
    python -m lem.live_test --grad-clip 0.5  # tighter gradient clipping

Reads from the existing tokenizer, converts TokenizedFrame to encoder inputs,
runs through the full training loop, and reports metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import time
from pathlib import Path

# Ensure Source/Python is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tokenizer import LEMTokenizer
from tokenizer.config import TokenizerConfig, MOD_AUDIO, MOD_PROPRIOCEPTION, MOD_MOTOR, MOD_TOUCH, MOD_VOICE

from lem.config.loader import load_and_resolve
from lem.model.encoder_registry import EncoderRegistry
from lem.model.forward_model import ForwardModel
from lem.model.prediction_heads import PredictionHeads, GazeHead, VoiceHead
from lem.model.motor_controller import MotorController
from lem.model.wellness import WellnessEvaluators, WellnessMode, MotorStateFormula
from lem.model.body_schema import DifferentiableBodySchema
from lem.model.body_sets import HANDLE_NAMES as _BODY_HANDLE_NAMES
from lem.model.receipt import emit_model_receipt
from lem.placeholders.system import PlaceholderSystem
from lem.training.loss_manager import LossManager
from lem.training.latent_regularizer import LatentRegularizer
from lem.source_mask import to_str as _src_to_str
from lem.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("live_test")

CONFIG_PATH = str(Path(__file__).resolve().parent / "master_config.yaml")
VALIDATION_DIR = Path(__file__).resolve().parent.parent / "Saved" / "Validation"


def report_handle_names(n_handles: int) -> list[str]:
    """Name of each handle row in the periodic report, taken from lem.model.body_sets.

    Row i of the report's handle accumulators is labelled with body handle i's name, so
    a row is named by its body part. The names come from the shared body_sets list (24
    names, the eyes "gaze" at row 23), never from a local copy: the 23-name copy this
    replaced ended at foot_R, so with the configured 24 handles the first report raised
    IndexError at the eyes' row and ended the run.

    A body with more rows than body_sets names - the MIMo trunk_v1 layout has 26, and
    the UE wire format carries 24 handle slots - is refused here at start-up, naming
    both counts, instead of failing inside the report.
    """
    if n_handles > len(_BODY_HANDLE_NAMES):
        raise ValueError(
            f"config body.n_handles is {n_handles}, but lem.model.body_sets names only "
            f"{len(_BODY_HANDLE_NAMES)} handles; the UE runner has no name for rows "
            f"{len(_BODY_HANDLE_NAMES)} to {n_handles - 1}")
    return list(_BODY_HANDLE_NAMES[:n_handles])


def apply_developmental_constraints(motor_out: torch.Tensor, dev_limits: dict,
                                    position_range_cm: float = 200.0) -> torch.Tensor:
    """Apply developmental phase limits to motor output for action writing.

    Motor output shape: [B, n_handles, 9] = pos(3) + rot(4) + radius(1) + strength(1).
    Position: clamp to [-1,1], scale by rom_limit, denormalize to UE cm.
    Strength: clamp to phase limit.  Rotation and radius pass through unchanged.

    The model operates in normalized [-1,1] space (tokenizer divides by position_range_cm).
    This function converts back to UE world-space centimeters for the action SHM write.

    Clones input to avoid mutating the caller's tensor (e.g. replay buffer storage).

    When curriculum is active, a handle the phase has not activated receives no
    position offset, no radius and no strength, and holds its orientation.
    """
    rom_limit = dev_limits.get("rom_limit", 1.0)
    strength_limit = dev_limits.get("strength_limit", 1.0)
    motor_out = motor_out.clone()
    # Clamp to [-1,1] (safety), scale by ROM fraction, denormalize to UE cm
    motor_out[..., :3] = motor_out[..., :3].clamp(-1.0, 1.0) * rom_limit * position_range_cm
    motor_out[..., 8:9].clamp_(0.0, strength_limit)            # Clamp strength

    # Handle masking (curriculum only)
    handle_mask = dev_limits.get("handle_mask")
    if handle_mask is not None:
        # lem/training/curriculum.py builds the mask from its own 24-entry HANDLE_NAMES
        # whatever the body carries; the MIMo trunk_v1 body has 26 handles. Callers pad
        # it with mimo_bridge.body_layout.extend_limits before the limits reach here
        # (mimo_bridge/lem_model.py dev_limits, train_mimo.py). Without that padding a
        # plain broadcast raises a shape error naming neither count.
        n_rows = motor_out.shape[-2]
        if len(handle_mask) != n_rows:
            raise ValueError(
                f"developmental handle_mask has {len(handle_mask)} entries against "
                f"{n_rows} motor rows - extend it for this body layout with "
                f"mimo_bridge.body_layout.extend_limits before applying limits")
        mask_t = torch.tensor(
            handle_mask, dtype=motor_out.dtype, device=motor_out.device)
        m = mask_t.unsqueeze(0).unsqueeze(-1)                  # [1, n_handles, 1]
        motor_out[..., 0:3] *= m                               # no position offset
        motor_out[..., 7:9] *= m                               # no radius, no strength
        # Channels 3:7 are a TARGET QUATERNION in xyzw order (lem_model.py gaze_to_handle
        # reads w as component 6), so identity is (0,0,0,1). Multiplying it by the mask
        # leaves (0,0,0,0), norm 0.0, which is not a rotation: every renormalisation on
        # this path clamps the norm (lem_model.smooth_blend, self_intention) so it yields
        # a zero rotation command rather than a NaN, but the masked handle is then
        # commanded to an orientation that does not exist instead of holding its own.
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                dtype=motor_out.dtype, device=motor_out.device)
        motor_out[..., 3:7] = motor_out[..., 3:7] * m + (1.0 - m) * identity

    return motor_out


def validate_step(step, tokenized, tokens, predictions,
                  motor_out, loss, all_params):
    """Dump diagnostics on first step to verify pipeline correctness."""
    if step != 0:
        return

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=== VALIDATION DUMP (step 1) ===")

    # 1. Vision data stats (GPU-side only, no transfer)
    if tokenized.vision_patches is not None:
        patches = tokenized.vision_patches  # [N, 16, 16, 3]
        N = patches.shape[0]
        log.info(f"  Vision: {N} patches, shape={list(patches.shape)}, "
                 f"min={patches.min():.4f}, max={patches.max():.4f}, "
                 f"mean={patches.mean():.4f}, std={patches.std():.4f}")
        if patches.std() < 1e-6:
            log.warning("  VISION DATA LOOKS CONSTANT (std < 1e-6)")
        # Save a small grid — only first 16 patches, minimal GPU→CPU
        try:
            import numpy as np
            small = patches[:min(16, N)].detach().float().cpu().numpy()
            cols = 4
            rows = (small.shape[0] + cols - 1) // cols
            grid = np.zeros((rows * 16, cols * 16, 3), dtype=np.float32)
            for i in range(small.shape[0]):
                r, c = divmod(i, cols)
                grid[r*16:(r+1)*16, c*16:(c+1)*16] = small[i]
            grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(grid).save(str(VALIDATION_DIR / "vision_patches_step1.png"))
            log.info(f"  Vision patches saved to {VALIDATION_DIR / 'vision_patches_step1.png'}")
        except Exception as e:
            log.warning(f"  Could not save vision patches: {e}")
    else:
        log.info("  Vision: no patches (not available)")

    # 2. CPU modality data ranges
    for name, tensor in [
        ("audio_spectral", tokenized.audio_spectral),
        ("proprio", tokenized.proprio),
        ("motor_handles", tokenized.motor_handles),
        ("touch_impulses", tokenized.touch_impulses),
    ]:
        if tensor is not None and tensor.numel() > 0:
            log.info(f"  {name}: shape={list(tensor.shape)}, "
                     f"min={tensor.min():.4f}, max={tensor.max():.4f}, "
                     f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")
        else:
            log.info(f"  {name}: empty/None")

    # 3. Encoder token outputs
    log.info("  Encoder tokens:")
    for mod, tok in tokens.items():
        log.info(f"    {mod}: shape={list(tok.shape)}, "
                 f"min={tok.min():.4f}, max={tok.max():.4f}, "
                 f"std={tok.std():.4f}, any_nan={tok.isnan().any().item()}")

    # 4. Prediction head outputs
    log.info("  Predictions:")
    for mod, pred in predictions.items():
        log.info(f"    {mod}: shape={list(pred.shape)}, "
                 f"min={pred.min():.4f}, max={pred.max():.4f}, "
                 f"std={pred.std():.4f}, any_nan={pred.isnan().any().item()}")

    # 5. Motor output
    if motor_out is not None:
        log.info(f"  Motor output: shape={list(motor_out.shape)}, "
                 f"min={motor_out.min():.4f}, max={motor_out.max():.4f}, "
                 f"std={motor_out.std():.4f}")

    # 6. Loss value
    log.info(f"  Loss: {loss.item():.6f}")

    # 7. Gradient check — use torch.nn.utils.clip_grad_norm_ style (single pass, no per-param .item())
    grads = [p.grad for p in all_params if p.grad is not None]
    n_with_grad = len(grads)
    n_total_params = len(all_params)
    if grads:
        total_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf')).item()
        log.info(f"  Gradient norm: {total_norm:.4f} ({n_with_grad}/{n_total_params} params have grad)")
    else:
        log.warning(f"  No gradients found! (0/{n_total_params} params)")

    log.info("=== END VALIDATION ===")


@contextlib.contextmanager
def _null_ctx():
    """No-op context manager (replacement for nullcontext on older Python)."""
    yield


def _to_device(tensor, target_device, non_blocking=True):
    """Transfer tensor to target device if needed."""
    if tensor is None:
        return None
    # Compare device type and index to handle torch.device('cuda:0') vs device(type='cuda', index=0)
    target = torch.device(target_device)
    if tensor.device.type == target.type and tensor.device.index == target.index:
        return tensor
    return tensor.to(target, non_blocking=non_blocking)


def _to_float(v):
    """Extract scalar from tensor or passthrough float."""
    return v.item() if isinstance(v, torch.Tensor) else float(v)


def _run_imagination_compiled(
    encoder_output_history, motor_history, imagination_steps,
    forward_model, prediction_heads, encoder_registry,
    context_buffer, active_modalities, n_modalities,
    latent_regularizer=None, latent_reg_weight=0.01,
    valid_history=None, achieved_state=None,
    target_history=None, pure_out=None, imagined_out=None, memory_bias=None,
):
    """Run imagination chain through COMPILED forward_model and prediction_heads.

    Unlike forward_model._orig_mod.imagine_step(), this calls the compiled
    (torch.compile) forward_model and prediction_heads directly. The motor
    encoder is tiny (~0.1ms) and accessed via _orig_mod — not worth the
    complexity of separate compilation.

    Returns imagination_loss (scalar tensor with live gradients).
    """
    registry_raw = getattr(encoder_registry, '_orig_mod', encoder_registry)

    imagined = dict(encoder_output_history[0])  # shallow copy of starting state
    imag_ctx = context_buffer.clone()  # real temporal context
    imag_losses = []

    for s in range(imagination_steps):
        motor_out = motor_history[s]
        B = motor_out.shape[0]

        # Encode the action in the SAME layout the live path uses, so the motor token the
        # forward model sees during imagination has the same meaning as the one it sees
        # awake. Live layout is [achieved(10), command(9)].
        #
        # achieved_state is the last REAL handle state. It is held across the rollout:
        # exact at depth 0, stale after that. Holding a real value keeps the token inside
        # the distribution the encoder was trained on; zeroing it would not. This is an
        # approximation and the depth-wise RMS log is what shows when it starts to hurt.
        # THE CANONICAL TRANSITION. This loop used to call the motor encoder module
        # directly and assemble the step by hand, so it skipped the registry's output
        # LayerNorm - the same defect as predict_from_action, in a route that never calls
        # predict_from_action and was therefore invisible to the AST check that proved
        # "no legacy call sites". One route, or the contract is a description rather than
        # a constraint.
        from lem.model.predictive_state import PredictiveState, transition
        # DOMAIN AND SOURCE MASK ARE NOT OPTIONAL HERE. transition() stamps
        # domain="predicted" on what it returns and this loop keeps only the tokens, so
        # a bare rebuild takes the dataclass defaults - domain "real" and an empty
        # source_mask - on states that hold the model's own predictions. That is the
        # invariant predictive_state.py was written to hold: "copying a real source mask
        # into an imagined state and leaving the domain implicit is how two
        # representations came to be scored as if they were one." Depth 0 is the real
        # starting frame; every depth after it is predicted.
        _sm = (valid_history[s] if valid_history is not None else None) or {}
        _st = PredictiveState(tokens={m: imagined[m] for m in active_modalities
                                      if m in imagined},
                              context=imag_ctx, achieved=achieved_state, depth=s, memory_bias=memory_bias,
                              domain=("real" if s == 0 else "predicted"),
                              source_mask={m: _src_to_str(v) for m, v in _sm.items()})
        _nx = transition(forward_model, registry_raw, prediction_heads, _st, motor_out,
                         achieved_state=achieved_state, advance_context=True)
        imag_ctx = _nx.context
        z_motor = _nx.tokens.get("motor")
        predicted = {m: v for m, v in _nx.tokens.items() if m != "motor"}
        if imagined_out is not None:
            # probe hook: the per-depth predictions, pre-anchoring, detached
            imagined_out.append({m: v.detach() for m, v in predicted.items()})

        # Loss vs ground truth encoder outputs.
        # Synthetic and stale targets are EXCLUDED at every depth. Without this the
        # prediction heads chase fresh Gaussian placeholder targets (audio, intermind)
        # once per rollout step at weight 0.5, which is unlearnable by construction and
        # pushes their outputs off the manifold the wellness scorer was trained on.
        # The mask is the one that was true at depth s+1, not the one at depth 0.
        # review M8 (2026-09-02): targets come from the RAW history when provided -
        # the observed history is the gated blend the live transformer consumed, and
        # grading against a blend that contains the model's own prediction lets the
        # chain partially grade itself. Anchoring (below) stays on the OBSERVED
        # history: the chain must be re-anchored to what it would have consumed.
        target = (target_history[s + 1] if target_history is not None
                  else encoder_output_history[s + 1])
        observed = encoder_output_history[s + 1]
        valid = valid_history[s + 1] if valid_history is not None else None
        graded = [m for m in predicted if m in target
                  and (valid is None or _src_to_str(valid.get(m, "real")) == "real")]
        # ADVANCE THE ROLLOUT BEFORE THE GRADING EXIT. imag_ctx was replaced by the
        # advanced context above; leaving `imagined` on the previous depth's state would
        # hand every later depth a state one hop behind its context, so a depth-2
        # prediction made from a depth-1 state gets graded against the depth-2 target -
        # a 2-step map taught under a 1-step label. See the FEED FORWARD note below for
        # why non-real channels are re-anchored to what was observed.
        imagined = dict(predicted)
        for m in list(imagined):
            if valid is not None and _src_to_str(valid.get(m, "real")) != "real":
                if m in observed:
                    imagined[m] = observed[m].detach()
        if not graded:
            if pure_out is not None:
                # hold the depth slot: without this the horizon license, which reads
                # pure_out by index-as-depth, credits depth d's result to depth d-1
                # whenever an earlier depth had no graded real sense
                pure_out.append(None)
            continue
        step_loss = sum(F.mse_loss(predicted[m], target[m]) for m in graded)
        if pure_out is not None:
            # the horizon license compares this against do-nothing; the latent
            # penalty added below must not bias that comparison
            pure_out.append(float(step_loss.detach()))
        if latent_regularizer is not None:
            # the same validity that filtered the target MSE also filters the latent
            # penalty: otherwise a synthetic channel is excluded from the lesson but
            # still shapes the latent distribution the lesson is regularised against
            step_loss = step_loss + latent_reg_weight * latent_regularizer.penalty(
                predicted, valid=valid)
        imag_losses.append(step_loss)
        # FEED FORWARD - and this is where H30 came apart on 2026-08-26.
        #
        # The loss above already excludes synthetic channels. The CHAIN did not: every
        # modality, real or fake, was fed back in as the next step's input. Audio and
        # inter-mind are fresh Gaussian placeholders on MIMo - unpredictable by
        # construction - so their predictions wander, get fed back, wander further, and
        # three steps of that lands far outside the distribution the encoders produce.
        # The forward model then reads those tokens alongside the real ones, so the
        # corruption reaches the channels the loss DOES grade.
        #
        # Measured on H30: inter-mind reconstruction error climbed 0.55 -> 200 (x340)
        # from the moment imagination was switched on, and once it dominated the loss
        # the real senses followed - vision x8, touch x7, body position x65 - while
        # next-state prediction, which never runs this chain, stayed flat. H22 ran
        # imagination with the placeholders OFF and H28 ran the placeholders with
        # imagination OFF; neither diverged. H30 was the first run with both.
        #
        # A channel the model cannot predict is carried forward from what was OBSERVED
        # rather than from what was imagined. The rollout stays anchored on those
        # channels instead of compounding noise into them. The assignment itself sits
        # ABOVE the grading exit so a depth with nothing to grade still advances.

    if not imag_losses:          # every depth fully masked: no gradient to give
        return None, []
    return sum(imag_losses), imag_losses


def _motor_encoder_width(encoder_registry):
    """How many features per handle the motor encoder actually wants.

    Read from the module rather than the config, because the two have disagreed: the
    config said 10 while LemLite overrode to 19, and later the defaults said 19 while
    master_config.yaml still said 10 and won the merge.
    """
    try:
        reg = getattr(encoder_registry, "_orig_mod", encoder_registry)
        return int(reg.encoders["motor"].patch_mlp[0].in_features)
    except Exception:
        return 0


def frame_receive_buffers(config, device):
    """The buffers build_frame_from_tokenized copies each frame's non-vision senses into.

    The UE runner (lem.inference_process) and the UE replay trainer (lem.replay_setup)
    build these once on CUDA, so a frame costs a .copy_() instead of an allocation.
    The body sense and touch rows are sized from the resolved config, the numbers their
    encoders are built from: the body sense is 12 numbers per bone
    (proprioception encoder _input_dim: 288 for the 24-bone body whose bone 23 is the
    eyes), touch is one row per skin patch with 10 fields (tactile pipeline n_sensors x
    features_per_sensor). Both used to be fixed, at 276 (23 bones) and 8 fields, and
    stopped matching when the eyes became bone 23 (2026-08-24) and when the two
    skin-stretch fields were delivered (2026-09-14): a 24-bone row and every touch row
    failed its copy.
    """
    active = config["_active_modalities"]
    n_handles = config["body"]["n_handles"]
    bufs = {
        "audio_spectral": torch.empty(1, 256, 2, device=device),
        "audio_spatial":  torch.empty(1, 2, device=device),
        "motor":          torch.empty(1, n_handles, 10, device=device),
        "voice":          torch.empty(1, 56, device=device),
        "motor_capability": torch.empty(21, device=device),
    }
    if "proprioception" in active:
        proprio_dim = int(config["modalities"]["proprioception"]["encoder"]["_input_dim"])
        bufs["proprio"] = torch.empty(1, proprio_dim, device=device)
    if "tactile" in active:
        tac = config["modalities"]["tactile"]["pipeline"]
        bufs["tactile"] = torch.empty(
            1, int(tac["n_sensors"]), int(tac["features_per_sensor"]), device=device)
    return bufs


def build_frame_from_tokenized(tokenized_frame, config, device,
                                xfer_stream=None, capture_device=None,
                                sync_transfers=False, vision_device=None,
                                timing_detail=None,
                                prealloc_patches=None, prealloc_angular=None,
                                prealloc_buffers=None, encoder_registry=None):
    """
    Convert a TokenizedFrame from the existing tokenizer into the format
    expected by EncoderRegistry.encode().

    Config parameters match tokenizer output shapes, so no padding or
    column extraction is needed — tensors pass through directly.

    When xfer_stream is provided (multi-GPU), transfers happen on that stream
    and a synchronize is issued before return.

    When sync_transfers is True (CUDA graph mode), transfers are synchronous
    on the default stream with no cross-device async dependencies.

    When vision_device is set (pipelined vision), vision patches stay on
    vision_device instead of being transferred to device. The vision encoder
    runs on vision_device and only the tiny output token crosses GPUs.

    When timing_detail is a dict, per-modality CPU timings (ms) are stored
    into it for stall diagnosis (overhead: ~6 perf_counter calls = <5µs).
    """
    frame = {}
    source_mask = {}
    active = config["_active_modalities"]
    mask = tokenized_frame.metadata.modality_valid_mask
    has_fresh_vision = False
    nb = not sync_transfers  # non_blocking flag for _to_device
    _t = timing_detail is not None

    # Optionally run transfers on dedicated stream for overlap
    stream_ctx = torch.cuda.stream(xfer_stream) if xfer_stream is not None else _null_ctx()

    with stream_ctx:
        # Vision — reshape on source device (free), then transfer.
        # With pipelined vision (vision_device set), patches stay on vision_device
        # for encoding there. Only the tiny [1,1,768] token crosses GPUs later.
        # Without pipelined vision, patches transfer to training device as before.
        if _t:
            _ts = time.perf_counter()
        if "vision" in active and tokenized_frame.vision_patches is not None:
            patches = tokenized_frame.vision_patches  # [N, 16, 16, 3] on capture_device
            N = patches.shape[0]
            flat = patches.reshape(N, -1).unsqueeze(0)  # [1, N, 768] — reshape is free
            if _t:
                timing_detail["b_vis_reshape"] = (time.perf_counter() - _ts) * 1000
                _ts = time.perf_counter()
            vis_target = vision_device if vision_device is not None else device
            if prealloc_patches is not None and prealloc_patches.shape[1] >= N:
                # Copy into pre-allocated buffer — no alloc/free cycle
                dst = prealloc_patches[:, :N]
                dst.copy_(flat, non_blocking=nb)
                flat = dst
            else:
                flat = _to_device(flat, vis_target, non_blocking=nb)
            if _t:
                timing_detail["b_vis_xfer"] = (time.perf_counter() - _ts) * 1000
                _ts = time.perf_counter()
            angular = tokenized_frame.vision_angular_pos
            if angular is not None:
                angular = angular.unsqueeze(0)  # [1, N, 2]
                if prealloc_angular is not None and prealloc_angular.shape[1] >= N:
                    dst_a = prealloc_angular[:, :N]
                    dst_a.copy_(angular, non_blocking=nb)
                    angular = dst_a
                else:
                    angular = _to_device(angular, vis_target, non_blocking=nb)
            frame["vision"] = {"patches": flat, "angular_pos": angular}
            source_mask["vision"] = "real"
            has_fresh_vision = True
            if _t:
                timing_detail["b_vis_rest"] = (time.perf_counter() - _ts) * 1000
        elif "vision" in active:
            source_mask["vision"] = "stale"
        if _t:
            timing_detail["b_vis"] = sum(timing_detail.get(k, 0)
                for k in ("b_vis_reshape", "b_vis_xfer", "b_vis_rest"))

        # Audio
        if _t:
            _ts = time.perf_counter()
        if "audio" in active and tokenized_frame.audio_spectral is not None and (mask & MOD_AUDIO):
            src_spectral = tokenized_frame.audio_spectral.unsqueeze(0)
            if prealloc_buffers and "audio_spectral" in prealloc_buffers:
                prealloc_buffers["audio_spectral"].copy_(src_spectral, non_blocking=nb)
                spectral = prealloc_buffers["audio_spectral"]
            else:
                spectral = _to_device(src_spectral, device, non_blocking=nb)
            meta = tokenized_frame.audio_metadata
            if meta is not None:
                src_spatial = meta[:2].unsqueeze(0)
                if prealloc_buffers and "audio_spatial" in prealloc_buffers:
                    prealloc_buffers["audio_spatial"].copy_(src_spatial, non_blocking=nb)
                    spatial = prealloc_buffers["audio_spatial"]
                else:
                    spatial = _to_device(src_spatial, device, non_blocking=nb)
            else:
                spatial = torch.zeros(1, 2, device=device)
            frame["audio"] = {"spectral": spectral, "spatial": spatial}
            source_mask["audio"] = "real"
        if _t:
            timing_detail["b_aud"] = (time.perf_counter() - _ts) * 1000

        # Proprioception — tokenizer outputs [n_bones, 12]; the encoder takes the config's
        # _input_dim (288 = 24 bones x 12, bone 23 = the eyes, in master_config.yaml)
        if _t:
            _ts = time.perf_counter()
        if "proprioception" in active and tokenized_frame.proprio is not None and (mask & MOD_PROPRIOCEPTION):
            src_proprio = tokenized_frame.proprio.reshape(-1).unsqueeze(0)
            _want = int(config["modalities"]["proprioception"]["encoder"]["_input_dim"])
            _got = int(src_proprio.shape[1])
            if _got != _want:
                _as_bones = lambda n: (f"{n // 12} bones x 12" if n % 12 == 0
                                       else "not a whole number of 12-number bones")
                raise ValueError(
                    f"The body sense row from the tokenizer has {_got} numbers "
                    f"({_as_bones(_got)}), but the config's proprioception encoder "
                    f"takes {_want} ({_as_bones(_want)}). The sender's bone list "
                    f"(UE C++ TrackedBoneNames) and the config's proprioception "
                    f"input_dim must name the same bones.")
            if prealloc_buffers and "proprio" in prealloc_buffers:
                prealloc_buffers["proprio"].copy_(src_proprio, non_blocking=nb)
                frame["proprioception"] = prealloc_buffers["proprio"]
            else:
                frame["proprioception"] = _to_device(src_proprio, device, non_blocking=nb)
            source_mask["proprioception"] = "real"

        # Motor — tokenizer outputs [n_handles, 10] physical fields; the encoder's
        # per_patch_input_dim (19) adds the command columns inside _encode_one
        if "motor" in active and tokenized_frame.motor_handles is not None and (mask & MOD_MOTOR):
            src_motor = tokenized_frame.motor_handles.unsqueeze(0)
            # frame["motor"] stays the PHYSICAL 10-field state. The encoder's extra
            # command columns are added inside EncoderRegistry._encode_one, so the motor
            # controller, the replay buffer, the shared ring and the wellness formulas all
            # keep the layout they were built for. Widening the frame here broke all four.
            if prealloc_buffers and "motor" in prealloc_buffers:
                prealloc_buffers["motor"].copy_(src_motor, non_blocking=nb)
                frame["motor"] = prealloc_buffers["motor"]
            else:
                frame["motor"] = _to_device(src_motor, device, non_blocking=nb)
            source_mask["motor"] = "real"

        # Tactile (touch) — TMC impulse sensor readings
        if "tactile" in active and tokenized_frame.touch_impulses is not None and (mask & MOD_TOUCH):
            src_tactile = tokenized_frame.touch_impulses.unsqueeze(0)
            if prealloc_buffers and "tactile" in prealloc_buffers:
                prealloc_buffers["tactile"].copy_(src_tactile, non_blocking=nb)
                frame["tactile"] = prealloc_buffers["tactile"]
            else:
                frame["tactile"] = _to_device(src_tactile, device, non_blocking=nb)
            source_mask["tactile"] = "real"
        if _t:
            timing_detail["b_rest"] = (time.perf_counter() - _ts) * 1000

        # Intermind — stub, always synthetic for now
        # (placeholder system will fill this)

        # Voice
        if "voice" in active and tokenized_frame.voice is not None and (mask & MOD_VOICE):
            src_voice = tokenized_frame.voice.unsqueeze(0)
            if prealloc_buffers and "voice" in prealloc_buffers:
                prealloc_buffers["voice"].copy_(src_voice, non_blocking=nb)
                frame["voice"] = prealloc_buffers["voice"]
            else:
                frame["voice"] = _to_device(src_voice, device, non_blocking=nb)
            source_mask["voice"] = "real"

    # Cross-device sync: ensure transfers complete before training stream uses data
    if xfer_stream is not None:
        xfer_stream.synchronize()

    return frame, source_mask, has_fresh_vision


def main():
    parser = argparse.ArgumentParser(description="LEM live training test against UE5")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--no-vision", action="store_true", help="Skip vision encoder (faster)")
    parser.add_argument("--precision", type=str, default=None, help="Override precision (fp16|bf16|fp32)")
    parser.add_argument("--report-every", type=int, default=10, help="Report interval")
    parser.add_argument("--timeout-ms", type=int, default=200, help="Frame read timeout (ms)")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N steps (0=end only, default: from config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (default: from config)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory path")
    parser.add_argument("--timing", action="store_true", help="Enable CUDA Event per-stage timing (adds sync overhead)")
    parser.add_argument("--warmup", type=int, default=20, help="Steps to exclude from final avg (JIT warmup)")
    parser.add_argument("--capture-device", type=str, default=None, help="Vision capture device (default: from config)")
    parser.add_argument("--training-device", type=str, default=None, help="Training device (default: from config)")
    parser.add_argument("--backward-every-n", type=int, default=None, help="Override backward_every_n_frames (1=every frame)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if config enables it")
    parser.add_argument("--stats-file", type=str, default=None,
                        help="Path for JSON Lines stats file (default: Saved/training_stats.jsonl)")
    parser.add_argument("--wellness-mode", type=str, default="baseline",
                        choices=["baseline", "direct", "forward"],
                        help="Wellness experiment mode (baseline|direct|forward)")
    parser.add_argument("--mse-weight", type=float, default=1.0,
                        help="Motor MSE weight for --wellness-mode forward")
    parser.add_argument("--wellness-grad-weight", type=float, default=1.0,
                        help="Wellness gradient weight for --wellness-mode forward")
    parser.add_argument("--motor-state-formula", type=str, default="preference",
                        choices=["preference", "boundary"],
                        help="Motor state wellness formula: preference (ROM center) or boundary (joint limits only)")
    parser.add_argument("--curiosity", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable curiosity system (default: from config)")
    parser.add_argument("--imagination", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable imagination-aware training (default: from config)")
    parser.add_argument("--latent-reg", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable latent regularizer (default: from config)")
    parser.add_argument("--adaptive-depth", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable adaptive depth motor controller (CF-7, default: from config)")
    parser.add_argument("--intention", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable intention generator (CF-9, default: from config)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")

    # Curriculum flags
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable 6-phase developmental curriculum (replaces static phases)")
    parser.add_argument("--start-phase", type=str, default="neonate",
                        choices=["neonate", "early_infant", "late_infant",
                                 "toddler", "child", "mature"],
                        help="Starting curriculum phase (default: neonate)")
    parser.add_argument("--steps-per-phase", type=int, default=50000,
                        help="Steps per curriculum phase (default: 50000)")
    parser.add_argument("--mastery-advancement", action="store_true",
                        help="Enable mastery-based early phase advancement")

    args = parser.parse_args()

    # Load and resolve config
    log.info(f"Loading config from {CONFIG_PATH}")
    config = load_and_resolve(CONFIG_PATH, skip_validation=True)

    if args.no_vision:
        config["modalities"]["vision"]["enabled"] = False
        from lem.config.resolver import resolve
        config = resolve(config)
        log.info("Vision disabled for faster testing")

    # Resolve device assignment (multi-GPU support)
    hw = config.get("hardware", {})
    capture_device = torch.device(
        args.capture_device or hw.get("capture_device", "cuda:0")
    ) if torch.cuda.is_available() else torch.device("cpu")
    training_device = torch.device(
        args.training_device or hw.get("training_device", args.device)
    ) if torch.cuda.is_available() else torch.device("cpu")
    multi_gpu = (capture_device != training_device)
    device = training_device  # Models live on training device

    # Resolve precision from config (or CLI override)
    precision_str = args.precision or config["model"].get("precision", "fp32")
    _PRECISION_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = _PRECISION_MAP.get(precision_str, torch.float32)
    if device.type != "cuda":
        dtype = torch.float32  # CPU doesn't benefit from fp16
    log.info(f"Training device: {training_device}, Capture device: {capture_device}, "
             f"multi_gpu: {multi_gpu}, precision: {precision_str} (dtype={dtype})")

    d_model = config["model"]["d_model"]
    active = config["_active_modalities"]
    log.info(f"Active modalities: {active}")
    log.info(f"d_model={d_model}, context_length={config['model']['context_length']}")

    # Initialize tokenizer FIRST — D3D12 interop must init on capture_device before
    # any model creation on training_device (cuda context ordering matters)
    log.info("Initializing tokenizer (connecting to UE SHM)...")
    try:
        tokenizer_config = TokenizerConfig()
        tokenizer_config.device = str(capture_device)
        tokenizer = LEMTokenizer(config=tokenizer_config)
    except Exception as e:
        log.error(f"Failed to initialize tokenizer: {e}")
        log.error("Make sure UE is running with the pipeline active.")
        sys.exit(1)
    log.info(f"Tokenizer ready on {capture_device}. Waiting for frames...")

    # Async frame reader (optional — background thread with ring buffer)
    async_cfg = config.get("async_reader", {})
    use_async = async_cfg.get("enabled", False)
    async_reader = None
    if use_async:
        from tokenizer.async_reader import AsyncFrameReader
        async_reader = AsyncFrameReader(
            tokenizer,
            ring_size=async_cfg.get("ring_size", 3),
            max_stale_age_ms=async_cfg.get("max_stale_age_ms", 50.0),
            # vision_every_n defaults to 1 (always capture). Reader must always
            # produce vision data so the training loop can encode it on its own
            # schedule (hardware.vision_every_n_frames controls encoding rate).
        )
        async_reader.start()

    # Dedicated transfer stream for cross-device copies (multi-GPU).
    # Disabled when using reduce-overhead (CUDA graphs) — async streams
    # create cross-device dependencies that invalidate graph capture.
    xfer_stream = None
    _tc = hw.get("torch_compile", {})
    use_cuda_graphs = _tc.get("enabled", False) and _tc.get("mode") == "reduce-overhead"
    if multi_gpu and training_device.type == "cuda" and not use_cuda_graphs:
        xfer_stream = torch.cuda.Stream(device=training_device)
        log.info(f"Cross-device transfer stream created on {training_device}")
    elif multi_gpu and use_cuda_graphs:
        log.info("CUDA graphs active — using synchronous transfers (no xfer_stream)")

    # Build model components (params stay fp32 — autocast handles fp16 forward pass)
    log.info("Building model components...")
    encoder_registry = EncoderRegistry(config).to(device)
    # Pipeline vision encoding: move vision encoder to capture GPU (cuda:1).
    # Eliminates 95 MB PCIe transfer per frame — only 3 KB token crosses GPUs.
    vision_enc_device = None
    if multi_gpu and capture_device is not None and capture_device != device:
        vision_enc_device = capture_device
        encoder_registry.move_vision_to(capture_device)
        log.info(f"Vision encoder pipelined to {capture_device} "
                 f"(primary: {device})")
    forward_model = ForwardModel(config).to(device)
    prediction_heads = PredictionHeads(config).to(device)
    motor_controller = MotorController(config).to(device)
    wellness_mode = WellnessMode(args.wellness_mode)
    motor_state_formula = MotorStateFormula(args.motor_state_formula)
    log.info(f"Wellness mode: {wellness_mode.value}, motor_state formula: {motor_state_formula.value}")
    wellness = WellnessEvaluators(config, wellness_mode=wellness_mode,
                                  motor_state_formula=motor_state_formula).to(device)
    body_schema = DifferentiableBodySchema(
        n_handles=config["body"]["n_handles"],
    )  # CPU — avoids allocator mutex
    # Pipelined body_schema motor: avoids .cpu() sync on default stream
    n_handles = config["body"]["n_handles"]
    _pinned_body_motor = torch.zeros(n_handles, 9, pin_memory=True) if device.type == "cuda" else None
    _body_motor_event = torch.cuda.Event() if device.type == "cuda" else None
    _pending_body_motor = None
    gaze_cfg = config["model"].get("gaze_head", {})
    gaze_head = None
    if gaze_cfg.get("enabled", False):
        log.info("Building GazeHead...")
        gaze_head = GazeHead(
            d_model=d_model,
            hidden_dim=gaze_cfg["hidden_dim"],
            n_heads=gaze_cfg["n_heads"],
            azimuth_range=gaze_cfg["azimuth_range"],
            elevation_range=gaze_cfg["elevation_range"],
        ).to(device)
    voice_cfg = config["model"].get("voice_head", {})
    voice_head = None
    if voice_cfg.get("enabled", False):
        log.info("Building VoiceHead...")
        voice_head = VoiceHead(
            d_model=d_model,
            hidden_dim=voice_cfg["hidden_dim"],
            n_heads=voice_cfg["n_heads"],
            output_dim=voice_cfg.get("output_dim", 80)).to(device)
    placeholder_system = PlaceholderSystem(config, device=str(device))
    loss_manager = LossManager(config, wellness_mode=args.wellness_mode)
    if args.wellness_mode == "forward":
        loss_manager.set_mode_weights(
            mse_weight=args.mse_weight,
            wellness_grad_weight=args.wellness_grad_weight,
        )
        log.info(f"FORWARD mode weights: mse={args.mse_weight}, wellness_grad={args.wellness_grad_weight}")

    # Curiosity system (UEMA Spec §21)
    curiosity_cfg = config.get("training", {}).get("curiosity", {})
    curiosity_enabled = (args.curiosity if args.curiosity is not None
                         else curiosity_cfg.get("enabled", True))
    curiosity_system = None
    if curiosity_enabled and wellness_mode != WellnessMode.BASELINE:
        from lem.model.curiosity import CuriositySystem
        curiosity_system = CuriositySystem(
            pe_min=curiosity_cfg.get("pe_min", 0.1),
            pe_max=curiosity_cfg.get("pe_max", 0.4))
        log.info(f"Curiosity system: pe_range=({curiosity_system.pe_min}, {curiosity_system.pe_max})")

    # Imagination-aware training (UEMA Spec §6.3)
    from collections import deque
    imagination_cfg = config.get("training", {}).get("imagination", {})
    imagination_enabled = (args.imagination if args.imagination is not None
                           else imagination_cfg.get("enabled", True))
    imagination_steps = imagination_cfg.get("steps", 3)
    imagination_every_n = imagination_cfg.get("every_n_steps", 1)
    # Silently disable if context_length too short
    if imagination_enabled and config["model"]["context_length"] < imagination_steps + 1:
        imagination_enabled = False
    encoder_output_history: deque = deque(maxlen=imagination_steps + 1)
    motor_history: deque = deque(maxlen=imagination_steps)
    # the source mask that was true at each depth. Without it the full LEM path grades
    # imagination against synthetic placeholder targets exactly as the bridge did.
    valid_history: deque = deque(maxlen=imagination_steps + 1)
    if imagination_enabled:
        log.info(f"Imagination training: steps={imagination_steps}, "
                 f"every_n={imagination_every_n}, weight={imagination_cfg.get('loss_weight', 0.5)}")

    # Latent regularizer (CF-11, UEMA Spec §6.3)
    latent_reg_cfg = config.get("training", {}).get("latent_reg", {})
    latent_reg_enabled = (args.latent_reg if args.latent_reg is not None
                          else latent_reg_cfg.get("enabled", True))
    latent_regularizer = None
    latent_reg_weight = latent_reg_cfg.get("loss_weight", 0.01)
    if latent_reg_enabled:
        latent_regularizer = LatentRegularizer(
            d_model=d_model,
            sigma_threshold=latent_reg_cfg.get("sigma_threshold", 2.0),
            momentum=latent_reg_cfg.get("momentum", 0.99),
        ).to(device)
        log.info(f"Latent regularizer: sigma={latent_regularizer.sigma_threshold}, "
                 f"momentum={latent_regularizer.momentum}, weight={latent_reg_weight}")

    # Intention generator (CF-9, UEMA Spec §10)
    intention_cfg = config.get("training", {}).get("intention", {})
    intention_enabled = (args.intention if args.intention is not None
                         else intention_cfg.get("enabled", False))
    intention_generator = None
    cross_sim_attention = None
    if intention_enabled:
        from lem.model.intention_generator import (
            IntentionGenerator, CrossSimulationAttention)
        intention_generator = IntentionGenerator(
            config, forward_model, encoder_registry, prediction_heads)
        cross_sim_attention = CrossSimulationAttention(config).to(device)
        log.info(f"Intention generator: n_steps={intention_cfg.get('n_steps', 50)}, "
                 f"lr={intention_cfg.get('lr', 0.1)}, "
                 f"max_plan_steps={intention_cfg.get('max_plan_steps', 5)}, "
                 f"cross_sim_rollouts={intention_cfg.get('cross_sim_rollouts', 5)}")

    # Adaptive depth motor controller (CF-7, UEMA Spec §9.1-9.3)
    mc_cfg = config["model"]["motor_controller"]
    adaptive_enabled = (args.adaptive_depth if args.adaptive_depth is not None
                        else mc_cfg.get("adaptive", False))
    if adaptive_enabled:
        log.info(f"Adaptive depth: max_depth={mc_cfg.get('max_depth', 6)}, "
                 f"threshold={mc_cfg.get('initial_depth_threshold', 0.7)}, "
                 f"energy_rate={mc_cfg.get('thinking_energy_rate', 0.001)}")

    # Receipt
    receipt = emit_model_receipt(config, encoder_registry, forward_model,
                                motor_controller, wellness, gaze_head=gaze_head,
                                voice_head=voice_head)

    # Optimizer
    all_params = (list(encoder_registry.parameters()) +
                  list(forward_model.parameters()) +
                  list(prediction_heads.parameters()) +
                  list(motor_controller.parameters()) +
                  list(wellness.parameters()))
    if gaze_head is not None:
        all_params += list(gaze_head.parameters())
    if voice_head is not None:
        all_params += list(voice_head.parameters())
    if cross_sim_attention is not None:
        all_params += list(cross_sim_attention.parameters())
    trainable_params = sum(p.numel() for p in all_params)
    log.info(f"Model built: {trainable_params:,} trainable params")
    train_cfg = config["training"]
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    opt_kwargs = dict(lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    use_fused = device.type == "cuda"
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(all_params, **opt_kwargs,
                                      betas=tuple(train_cfg["betas"]),
                                      fused=use_fused)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(all_params, **opt_kwargs,
                                     betas=tuple(train_cfg["betas"]),
                                     fused=use_fused)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(all_params, **opt_kwargs,
                                    momentum=train_cfg["betas"][0])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    warmup_steps = train_cfg.get("warmup_steps", 1000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    )
    log.info(f"LR warmup: {warmup_steps} steps (linear ramp to {train_cfg['lr']})")

    # Mixed-precision: AMP autocast for fp16/bf16; GradScaler only for fp16.
    # bf16 has sufficient dynamic range (8-bit exponent) and doesn't need loss scaling.
    # Removing the scaler eliminates 2 GPU→CPU sync stalls per step (unscale_ + step).
    use_amp = dtype != torch.float32 and device.type == "cuda"
    use_scaler = use_amp and dtype == torch.float16  # scaler only needed for fp16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=1024)
    log.info(f"AMP: {'enabled' if use_amp else 'disabled'} (dtype={precision_str}, "
             f"scaler={'active' if use_scaler else 'off — bf16 needs no scaling'})")

    # Reconcile checkpoint CLI with config YAML (CLI wins when explicitly provided)
    _ckpt_cfg = config.get("checkpoint", {})
    if args.save_every is None:
        args.save_every = _ckpt_cfg.get("save_every_steps", 5000)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = _ckpt_cfg.get("checkpoint_dir", "checkpoints")

    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(config, checkpoint_dir=args.checkpoint_dir)
    start_step = 0
    if args.resume:
        log.info(f"Resuming from checkpoint: {args.resume}")
        load_result = checkpoint_mgr.load(
            args.resume, encoder_registry, forward_model,
            prediction_heads, motor_controller, wellness,
            optimizer, gaze_head=gaze_head,
            voice_head=voice_head,
            body_schema=body_schema,
            map_location=device,
        )
        start_step = load_result["step"]
        if "scaler_state" in load_result:
            scaler.load_state_dict(load_result["scaler_state"])
        log.info(f"Resumed at step {start_step}: loaded={load_result['loaded_components']}, "
                 f"reinit={load_result['reinitialized_components']}")

    # Curriculum manager (optional — replaces static phase lookup)
    curriculum_mgr = None
    if args.curriculum:
        from lem.training.curriculum import CurriculumManager
        curriculum_mgr = CurriculumManager(
            steps_per_phase=args.steps_per_phase,
            mastery_advancement=args.mastery_advancement,
            start_phase=args.start_phase,
        )
        if args.resume and "curriculum_state" in load_result:
            curriculum_mgr.load_state_dict(load_result["curriculum_state"])
            log.info(f"Curriculum restored: phase={curriculum_mgr.phase_name}")
        # Override imagination/context from curriculum phase
        phase_def = curriculum_mgr.phase_def
        imagination_enabled = phase_def.imagination_enabled
        imagination_steps = phase_def.imagination_steps
        config["model"]["context_length"] = phase_def.context_length
        log.info(f"Curriculum override: imagination={'on' if imagination_enabled else 'off'}, "
                 f"context_length={phase_def.context_length}")

    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
        log.info(f"VRAM after model load ({device}): {alloc_mb:.0f} MB allocated, {reserved_mb:.0f} MB reserved")
        if multi_gpu:
            cap_alloc = torch.cuda.memory_allocated(capture_device) / 1024**2
            log.info(f"VRAM capture device ({capture_device}): {cap_alloc:.0f} MB allocated")
            if torch.cuda.is_available() and capture_device.type == "cuda" and training_device.type == "cuda":
                can_p2p = torch.cuda.can_device_access_peer(training_device.index or 0,
                                                            capture_device.index or 0)
                log.info(f"P2P access {capture_device} → {training_device}: {can_p2p}")

    # SDPA backend status (FlashAttention detection)
    if device.type == "cuda":
        flash = torch.backends.cuda.flash_sdp_enabled()
        mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
        log.info(f"SDPA backends: flash={flash}, mem_efficient={mem_eff}")

    # torch.compile (Phase 4 — kernel fusion on Blackwell Tensor Cores)
    compile_cfg = hw.get("torch_compile", {})
    if compile_cfg.get("enabled", False) and device.type == "cuda" and not args.no_compile:
        compile_mode = compile_cfg.get("mode", "default")
        log.info(f"Applying torch.compile(mode='{compile_mode}') to model components...")
        encoder_registry = torch.compile(encoder_registry, mode=compile_mode, fullgraph=False)
        forward_model = torch.compile(forward_model, mode=compile_mode, fullgraph=False)
        # Skip compiling motor_controller — its metadata dict construction
        # (.detach(), conditional branching) breaks CUDAGraphs in reduce-overhead.
        prediction_heads = torch.compile(prediction_heads, mode=compile_mode, fullgraph=False)
        if gaze_head is not None:
            gaze_head = torch.compile(gaze_head, mode=compile_mode, fullgraph=False)
        if voice_head is not None:
            voice_head = torch.compile(voice_head, mode=compile_mode, fullgraph=False)
        log.info("torch.compile applied (first few steps will be slower due to JIT warmup)")

    # Action buffer writer (optional — writes motor/gaze to SHM for UE)
    action_writer = None
    if config.get("action_buffer", {}).get("enabled", False):
        from lem.action.writer import ActionBufferWriter
        action_writer = ActionBufferWriter(config)
        log.info("Action buffer writer active (SHM: %s)",
                 config["action_buffer"]["shm_name"])

    # Ensure default CUDA device matches training device — required for
    # CUDA graph capture (reduce-overhead) when tokenizer init set it to capture_device
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Training loop
    step = start_step
    last_vision_token = None
    last_tokens = {}  # General stale token cache for any modality missing from mask
    total_loss_accum = torch.zeros(1, device=device)  # GPU-side loss accumulator (avoids .item() sync)
    frame_times = []
    no_frame_count = 0
    max_no_frame = 500  # Bail after this many consecutive empty reads

    # Context buffer — rolling window of (ctx-1) previous frames for temporal attention.
    # Initialized to zeros (static shape for torch.compile); fills with real data as frames arrive.
    ctx_len = config["model"]["context_length"]
    n_mod = len(active)
    context_buffer = torch.zeros(1, (ctx_len - 1) * n_mod, d_model, device=device)

    # Pre-allocated buffers — avoid per-step tensor allocation overhead
    n_handles = config["body"]["n_handles"]
    features_per_handle = config["modalities"]["motor"]["pipeline"]["features_per_handle"]
    gaze_target_buf = torch.zeros(1, 2, device=device, dtype=torch.float32) if gaze_head is not None else None
    handle_state_fallback = torch.zeros(1, n_handles, features_per_handle, device=device)

    # ── Stats accumulators (GPU-side where possible, no per-step .item() calls) ──
    # Loss breakdown accumulators
    loss_breakdown_accum = {}
    for mod in config.get("_prediction_head_modalities", []):
        loss_breakdown_accum[f"pred_{mod}"] = torch.zeros(1, device=device)
    for key in ("motor", "wellness", "gaze", "vicreg", "motor_mse", "motor_wellness",
                "imagination", "latent_reg"):
        loss_breakdown_accum[key] = torch.zeros(1, device=device)

    # Motor confidence accumulators
    confidence_sum = torch.zeros(1, device=device)
    confidence_min = torch.ones(1, device=device)
    confidence_max = torch.zeros(1, device=device)

    # Motor depth accumulators (CF-7) — counts ALL motor controller calls
    # (training + perception-only) so depth_avg reflects true usage distribution.
    motor_depth_accum = 0
    motor_depth_min = mc_cfg.get("max_depth", 6)
    motor_depth_max = 0
    motor_depth_count = 0

    # Handle utilization accumulators
    handle_rom_accum = torch.zeros(n_handles, device=device)
    handle_str_accum = torch.zeros(n_handles, device=device)
    HANDLE_NAMES = report_handle_names(n_handles)  # from body_sets; refuses unnamed rows now

    # Encoder activation accumulators
    enc_act_mean_accum = {mod: torch.zeros(1, device=device) for mod in active}
    enc_act_std_accum = {mod: torch.zeros(1, device=device) for mod in active}

    # Wellness diagnostic accumulators
    wellness_component_accum = {
        name: torch.zeros(1, device=device)
        for name in ["prediction_error", "tactile_comfort", "capability", "coherence", "motor_state"]
    }
    wellness_total_accum = torch.zeros(1, device=device)
    motor_health_mean_accum = torch.zeros(1, device=device)
    motor_health_std_accum = torch.zeros(1, device=device)
    motor_health_min_accum = torch.full((1,), float('inf'), device=device)
    motor_health_max_accum = torch.full((1,), float('-inf'), device=device)
    motor_health_diversity_accum = torch.zeros(1, device=device)

    # Per-encoder timing accumulator (CPU-side, wall-clock ms)
    encoder_timing_accum = {mod: 0.0 for mod in active}

    # Async reader interval stats
    last_async_stats = {"captured": 0, "dropped": 0, "throttle_sleeps": 0}

    # JSON Lines stats file
    stats_file_path = args.stats_file or str(ROOT / "Saved" / "training_stats.jsonl")
    Path(stats_file_path).parent.mkdir(parents=True, exist_ok=True)
    stats_file = open(stats_file_path, "a")
    log.info(f"Stats file: {stats_file_path}")

    # GPU info — emitted once in first stats record
    _gpu_info = None
    _gpu_info_emitted = False
    if device.type == "cuda":
        _gprops = torch.cuda.get_device_properties(device)
        _driver_mode = "TCC" if not _gprops.is_integrated else "WDDM"
        _gpu_info = {
            "name": _gprops.name,
            "total_vram_mb": round(_gprops.total_memory / (1024**2), 0),
            "sm_count": _gprops.multi_processor_count,
            "compute_capability": f"{_gprops.major}.{_gprops.minor}",
            "driver_mode": _driver_mode,
        }

    # Transfer timing accumulator (averaged over report interval)
    _xfer_timing_accum: dict[str, float] = {}
    _xfer_timing_count = 0

    # CUDA Event timing instrumentation (Phase 0)
    use_timing = args.timing and device.type == "cuda"
    if use_timing:
        evt_start  = torch.cuda.Event(enable_timing=True)
        evt_read   = torch.cuda.Event(enable_timing=True)
        evt_xfer   = torch.cuda.Event(enable_timing=True)
        evt_encode = torch.cuda.Event(enable_timing=True)
        evt_fwd    = torch.cuda.Event(enable_timing=True)
        evt_motor  = torch.cuda.Event(enable_timing=True)
        evt_loss   = torch.cuda.Event(enable_timing=True)
        evt_bwd    = torch.cuda.Event(enable_timing=True)
        evt_step   = torch.cuda.Event(enable_timing=True)
        # Accumulate per-stage times for averaging
        timing_accum = {"read": 0.0, "xfer": 0.0, "enc": 0.0, "fwd": 0.0,
                        "motor": 0.0, "loss": 0.0, "bwd": 0.0, "optim": 0.0, "total": 0.0}
        timing_count = 0
        log.info("CUDA Event timing enabled (adds synchronize overhead per step)")

    # Async backward: forward every frame for motor commands, backward only every Nth
    batch_size = config["training"].get("batch_size", 1)
    backward_every_n = args.backward_every_n or config["training"].get("backward_every_n_frames", 1)
    frame_counter = 0  # Increments every frame (perception + training)
    perception_times = []  # Track perception-only frame times separately
    vision_every_n = config.get("hardware", {}).get("vision_every_n_frames", 1)
    vision_encode_count = 0  # Track how many frames ran vision encoder
    from lem.training.harness import _get_developmental_phase
    grad_clip = args.grad_clip
    grad_norm_accum = 0.0
    grad_norm_count = 0

    if backward_every_n > 1:
        log.info(f"backward_every_n={backward_every_n}: training every {backward_every_n} frames "
                 f"({backward_every_n - 1} perception-only frames between training steps)")

    # Async backward: overlap backward with next frame's forward on separate stream.
    # Only useful when backward_every_n > 1 (perception frames provide overlap window).
    # backward() writes .grad only (doesn't modify params), so it's safe to overlap
    # with forward reads of the same params. optimizer.step() is deferred until the
    # backward completes (synced before next training frame's forward).
    async_backward = hw.get("async_backward", False) and backward_every_n > 1
    backward_stream = None
    pending_backward = False
    deferred_loss = None  # Keep loss tensor alive until async backward completes
    deferred_grad_clip = grad_clip  # Snapshot for deferred clip
    if async_backward and device.type == "cuda":
        backward_stream = torch.cuda.Stream(device=device)
        log.info(f"Async backward enabled: backward on separate CUDA stream, "
                 f"optimizer.step() deferred to last perception frame")

    # Decoupled imagination backward: run imagination loss backward on a separate
    # CUDA stream so it overlaps with perception frames. Mathematically equivalent
    # to single backward (same gradient accumulation). Requires backward_every_n > 1
    # so perception frames exist to overlap with.
    imagination_stream = None
    pending_imagination = False
    imag_bwd_done_event = None  # GPU-side guard: prevents CUDAGraph replay while imag backward runs
    if imagination_enabled and device.type == "cuda" and backward_every_n > 1:
        imagination_stream = torch.cuda.Stream(device=device)
        log.info("Imagination backward decoupled: async on separate CUDA stream")

    if vision_every_n > 1:
        log.info(f"Split-rate vision: encoding every {vision_every_n} frames "
                 f"(~{60 // vision_every_n}Hz equivalent)")

    if not config["training"]["loss"].get("vicreg", {}).get("enabled", False):
        log.warning("InfoNCE contrastive loss not implemented — "
                    "encoder collapse prevention relies on prediction diversity only")

    mean_pe = 0.0  # Updated each step when curiosity_system is active

    end_step = start_step + args.steps
    try:
        while step < end_step:
            t0 = time.perf_counter()
            if use_timing:
                evt_start.record()

            # Read frame from UE (async or blocking)
            is_stale_frame = False
            capture_event = None  # Set by async reader if available
            if use_async:
                result = async_reader.get_latest_frame()
                if result is None:
                    no_frame_count += 1
                    if no_frame_count >= max_no_frame:
                        log.warning(f"No frames from async reader for {max_no_frame} iterations. "
                                    f"Is UE running?")
                        no_frame_count = 0
                    time.sleep(0.001)
                    continue
                tokenized, is_stale_frame, capture_event = result
            else:
                tokenized = tokenizer.read_frame_blocking(timeout_ms=args.timeout_ms)
                if tokenized is None:
                    no_frame_count += 1
                    if no_frame_count >= max_no_frame:
                        log.warning(f"No frames for {max_no_frame} consecutive reads. "
                                    f"Is UE running?")
                        no_frame_count = 0
                    continue

            no_frame_count = 0
            frame_counter += 1
            is_training_frame = (backward_every_n <= 1) or (frame_counter % backward_every_n == 0)

            # Resolve any pending async backward before next forward pass.
            # For perception frames: sync early (backward likely finished during prev frame).
            # For training frames: ensures previous backward + optimizer.step() complete
            # before this frame's forward reads the updated parameters.
            if pending_backward:
                backward_stream.synchronize()
                if use_scaler:
                    scaler.unscale_(optimizer)
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, deferred_grad_clip, foreach=True, error_if_nonfinite=False)
                grad_norm_accum += pre_clip_norm.item()
                grad_norm_count += 1
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                deferred_loss = None
                pending_backward = False

            # Resolve pending decoupled imagination backward.
            # Only resolve at training frame boundaries — perception frames don't need
            # updated params (everything is detached under no_grad). This lets the
            # imagination backward overlap with ALL perception frames instead of
            # blocking the first one.
            if pending_imagination and is_training_frame:
                imagination_stream.synchronize()
                if use_scaler:
                    scaler.unscale_(optimizer)
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, deferred_grad_clip, foreach=True, error_if_nonfinite=False)
                grad_norm_accum += pre_clip_norm.item()
                grad_norm_count += 1
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                deferred_loss = None
                pending_imagination = False

            t_read = time.perf_counter()
            if use_timing:
                evt_read.record()

            # Ensure tokenizer's CUDA work on capture device is complete before
            # transferring tensors to training device. GPU-side event wait avoids
            # blocking the CPU — the GPU command queue waits, CPU dispatches freely.
            if multi_gpu and use_cuda_graphs and capture_device.type == "cuda":
                if use_async and capture_event is not None:
                    torch.cuda.current_stream(device).wait_event(capture_event)
                else:
                    torch.cuda.synchronize(capture_device)  # Fallback: full sync

            # Convert to encoder format (transfers to training_device if multi-GPU).
            # With pipelined vision, patches stay on vision_enc_device for encoding there.
            _build_detail = {}
            frame, source_mask, has_fresh_vision = build_frame_from_tokenized(
                tokenized, config, device,
                xfer_stream=xfer_stream, capture_device=capture_device,
                sync_transfers=(multi_gpu and use_cuda_graphs),
                vision_device=vision_enc_device,
                timing_detail=_build_detail if device.type == "cuda" else None, encoder_registry=encoder_registry)

            # Accumulate transfer timings for dashboard
            if _build_detail:
                for _k, _v in _build_detail.items():
                    _xfer_timing_accum[_k] = _xfer_timing_accum.get(_k, 0.0) + _v
                _xfer_timing_count += 1

            # Fill synthetic modalities
            synthetic_entries = placeholder_system.fill_frame(frame, batch_size=batch_size)
            source_mask.update(synthetic_entries)

            # Mark remaining as real
            for mod in active:
                if mod not in source_mask:
                    source_mask[mod] = "real"

            # Handle stale vision
            if not has_fresh_vision and last_vision_token is not None:
                source_mask["vision"] = "stale"

            # Async reader: mark all modalities stale if frame is reused
            if is_stale_frame:
                for mod in active:
                    if source_mask.get(mod) == "real":
                        source_mask[mod] = "stale"

            # Split-rate vision: skip vision encoder on non-vision frames
            run_vision = (frame_counter % vision_every_n == 0) or (last_vision_token is None)
            if not run_vision and last_vision_token is not None:
                frame.pop("vision", None)
                source_mask["vision"] = "stale"
                has_fresh_vision = False
            if has_fresh_vision:
                vision_encode_count += 1

            if use_timing:
                evt_xfer.record()

            # Developmental phase limits (shared by both paths)
            if curriculum_mgr is not None:
                dev_limits = curriculum_mgr.get_dev_limits()
            else:
                dev_limits = _get_developmental_phase(config, step)

            # Get handle state (shared by both paths) — use pre-allocated fallback
            handle_state = frame.get("motor", handle_state_fallback)
            if handle_state.dim() == 2:
                handle_state = handle_state.unsqueeze(0)

            # Tell CUDAGraphs this is a new step — prevents buffer reuse from
            # previous replay overwriting tensors we still reference (e.g. motor_history).
            torch.compiler.cudagraph_mark_step_begin()

            # GPU-side guard: ensure imagination backward has finished before any
            # CUDAGraph replay overwrites activation buffers it may still be reading.
            # Uses wait_event (GPU-side only, zero CPU block) instead of synchronize().
            if pending_imagination and imag_bwd_done_event is not None:
                torch.cuda.current_stream().wait_event(imag_bwd_done_event)

            if is_training_frame:
                # ============================================================
                # TRAINING FRAME: full forward + backward + optimizer step
                # ============================================================
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    if step < 3 and multi_gpu:
                        log.info(f"[DBG] step={step} frame keys={list(frame.keys())} sources={dict(source_mask)}")
                        for k, v in frame.items():
                            if isinstance(v, torch.Tensor):
                                log.info(f"[DBG]   {k}: shape={v.shape} device={v.device} dtype={v.dtype}")
                            elif isinstance(v, dict):
                                for kk, vv in v.items():
                                    if isinstance(vv, torch.Tensor):
                                        log.info(f"[DBG]   {k}.{kk}: shape={vv.shape} device={vv.device} dtype={vv.dtype}")

                    tokens, enc_timings = encoder_registry.encode_timed(frame, source_mask)
                    for mod_t, ms_t in enc_timings.items():
                        encoder_timing_accum[mod_t] = encoder_timing_accum.get(mod_t, 0.0) + ms_t

                    if step < 3 and multi_gpu:
                        log.info(f"[DBG] encoder done, tokens={list(tokens.keys())}")

                    # Handle stale vision token reuse
                    if source_mask.get("vision") == "stale" and last_vision_token is not None:
                        tokens["vision"] = last_vision_token
                    elif "vision" in tokens:
                        last_vision_token = tokens["vision"].detach()

                    # Handle stale token reuse for any modality missing from mask.
                    # Pipeline now finalizes unconditionally (no modality gate), so
                    # any async/late modality may be absent. Cache last real token.
                    for mod in active:
                        if mod == "vision":
                            continue  # Vision handled above with its own cache
                        if source_mask.get(mod) == "real" and mod in tokens:
                            last_tokens[mod] = tokens[mod].detach()
                        elif source_mask.get(mod) != "real" and mod in last_tokens:
                            tokens[mod] = last_tokens[mod]

                    # Normalize token dict: fixed key set AND order so
                    # torch.compile sees one single specialization.
                    _zero = torch.zeros(1, 1, d_model, device=device)
                    tokens = {m: tokens.get(m, _zero) for m in active}

                    # Append encoder outputs to imagination history buffer
                    if imagination_enabled:
                        enc_snap = {mod: tokens[mod].squeeze(1).detach()
                                    for mod in tokens}
                        encoder_output_history.append(enc_snap)
                        valid_history.append(dict(source_mask))

                    # Update latent regularizer with real encoder outputs
                    if latent_regularizer is not None:
                        latent_regularizer.update({mod: tokens[mod].squeeze(1).detach()
                                                   for mod in tokens}, valid=source_mask)

                    # Accumulate encoder activation stats (GPU-side, no .item())
                    with torch.no_grad():
                        for mod_a, tok_a in tokens.items():
                            enc_act_mean_accum[mod_a] += tok_a.mean()
                            enc_act_std_accum[mod_a] += tok_a.std()

                    # GPU-side sync: ensure pipelined vision token is ready on cuda:0
                    # before forward model consumes it. Zero CPU overhead.
                    if vision_enc_device is not None:
                        encoder_registry_raw = getattr(
                            encoder_registry, '_orig_mod', encoder_registry)
                        encoder_registry_raw.sync_vision()

                    t_encode = time.perf_counter()
                    if use_timing:
                        evt_encode.record()

                    latents, last_step, ctx_update = forward_model(tokens, context_buffer)
                    if step < 3 and multi_gpu:
                        log.info(f"[DBG] forward_model done")
                    predictions = prediction_heads(last_step)
                    if step < 3 and multi_gpu:
                        log.info(f"[DBG] prediction_heads done")

                    if use_timing:
                        evt_fwd.record()

                    # --- Wellness gradient computation (UEMA Spec §7.4) ---
                    # Compute targets for prediction heads
                    targets = {mod: tokens[mod].squeeze(1).detach()
                               for mod in config["_prediction_head_modalities"]
                               if mod in tokens}

                    # Update developmental limits for formula evaluators
                    if wellness_mode != WellnessMode.BASELINE:
                        wellness.set_developmental_limits(dev_limits["rom_limit"])

                    # Create motor commands leaf with grad tracking for wellness path
                    motor_commands = handle_state[:, :, :9].detach().clone().requires_grad_(True)

                    # Compute wellness BEFORE motor controller (feel-before-act)
                    # Body schema coherence from previous frame's update (one-frame lag, fine with EMA)
                    bs_coherence = body_schema.coherence_score()
                    wellness_out = wellness(predictions, targets, latents,
                                           capability_features=tokenized.motor_capability,
                                           motor_commands=motor_commands,
                                           body_schema_coherence=bs_coherence,
                                           raw_tactile=frame.get("tactile"))

                    # Compute gradient: d(total_wellness) / d(motor_commands)
                    total_wellness = torch.cat(
                        list(wellness_out["grounded"].values()), dim=-1
                    ).sum()
                    wellness_grad = torch.autograd.grad(
                        total_wellness, motor_commands,
                        create_graph=True, retain_graph=True
                    )[0]  # [B, n_handles, 9]

                    # Motor controller with wellness gradient input
                    motor_out, motor_confidence, motor_meta = motor_controller(
                        latents, handle_state, wellness_grad=wellness_grad)

                    # CF-7 §9.2: Iterative deepening — inference-time planning only.
                    # During training, the soft-weighted residual blocks in the adaptive
                    # forward pass are the differentiable equivalent of hard selection.
                    # Spec §9.2 lines 210-226 pseudocode doesn't distinguish train/eval,
                    # but the soft weighting achieves the same gradient signal without the
                    # 6x compute multiplier that would hit nearly every neonate frame.
                    if (adaptive_enabled
                            and imagination_enabled
                            and not forward_model.training
                            and motor_meta.get("confidence", 1.0) < motor_meta.get("depth_threshold", 0.7)):
                        target_depth = motor_meta["depth"]
                        available_energy = (tokenized.motor_capability[-1].item()
                                            if tokenized.motor_capability is not None else 1.0)
                        best_wellness = -float('inf')
                        best_motor = motor_out
                        forward_model_raw = getattr(forward_model, '_orig_mod', forward_model)
                        for d in range(1, target_depth + 1):
                            if d * mc_cfg.get("thinking_energy_rate", 0.001) > available_energy * 0.1:
                                break
                            motor_d, _, _ = motor_controller(
                                latents, handle_state, wellness_grad=wellness_grad, depth=d)
                            # NOTE: Spec §9.2 line 217 uses forward_model.rollout(state,
                            # candidate, steps=depth) — deeper thinking = more imagination
                            # steps. We start with single-step predict_from_action since
                            # multi-step rollout requires specifying intermediate motor
                            # actions (underspecified in spec). Multi-step is the target.
                            predicted_d, z_motor_d = forward_model_raw.predict_from_action(
                                last_step, motor_d, encoder_registry, prediction_heads,
                                state_tokens={m: tokens[m] for m in tokens},
                                context_buffer=context_buffer)
                            w_d = wellness.score_predicted_wellness(predicted_d, z_motor_d).sum()
                            if w_d.item() > best_wellness:
                                best_wellness = w_d.item()
                                best_motor = motor_d
                        motor_out = best_motor

                    # Append motor output to imagination history buffer
                    if imagination_enabled:
                        motor_history.append(motor_out.detach())

                    # Accumulate motor depth stats (CF-7)
                    _step_depth = motor_meta.get("depth", mc_cfg["n_layers"])
                    motor_depth_accum += _step_depth
                    motor_depth_min = min(motor_depth_min, _step_depth)
                    motor_depth_max = max(motor_depth_max, _step_depth)
                    motor_depth_count += 1

                    # Accumulate motor confidence + handle utilization (GPU-side)
                    with torch.no_grad():
                        confidence_sum += motor_confidence.detach().mean()
                        confidence_min = torch.min(confidence_min, motor_confidence.detach().min())
                        confidence_max = torch.max(confidence_max, motor_confidence.detach().max())
                        handle_rom_accum += motor_out[0, :, :3].detach().norm(dim=-1)
                        handle_str_accum += motor_out[0, :, 8].detach()

                    # Accumulate wellness diagnostics (GPU-side)
                    with torch.no_grad():
                        for wname, wscore in wellness_out["grounded"].items():
                            wellness_component_accum[wname] += wscore.detach().mean()
                        wellness_total_accum += total_wellness.detach()
                        motor_health_mean_accum += motor_out.detach().mean()
                        motor_health_std_accum += motor_out.detach().std()
                        motor_health_min_accum = torch.min(
                            motor_health_min_accum, motor_out.detach().min().unsqueeze(0))
                        motor_health_max_accum = torch.max(
                            motor_health_max_accum, motor_out.detach().max().unsqueeze(0))
                        handle_norms = motor_out[0, :, :3].detach().norm(dim=-1)  # [n_handles]
                        motor_health_diversity_accum += handle_norms.std().unsqueeze(0)

                    # FORWARD mode: predict sensory consequences of motor action
                    forward_wellness_loss = None
                    if wellness_mode == WellnessMode.FORWARD:
                        predicted_tokens, z_motor = forward_model.predict_from_action(
                            last_step, motor_out, encoder_registry, prediction_heads,
                            state_tokens={m: tokens[m] for m in tokens},
                            context_buffer=context_buffer)
                        w_predicted = wellness.score_predicted_wellness(
                            predicted_tokens, z_motor)
                        forward_wellness_loss = -w_predicted.sum()

                    gaze_out = None
                    gaze_target = None
                    if gaze_head is not None:
                        gaze_out = gaze_head(last_step)
                        gaze_az_rad = math.radians(tokenized.metadata.gaze_azimuth_deg)
                        gaze_el_rad = math.radians(tokenized.metadata.gaze_elevation_deg)
                        gaze_target_buf[0, 0] = gaze_az_rad
                        gaze_target_buf[0, 1] = gaze_el_rad
                        gaze_target = gaze_target_buf

                    voice_out = None
                    if voice_head is not None:
                        voice_out = voice_head(last_step)

                    # Update body schema (UEMA Spec §11) — pipelined, one frame stale
                    if _pending_body_motor is not None and tokenized.proprio is not None:
                        body_schema.update(_pending_body_motor, tokenized.proprio)
                    if _pinned_body_motor is not None:
                        _body_motor_event.synchronize()
                        _pinned_body_motor.copy_(motor_out.detach().squeeze(0), non_blocking=True)
                        _body_motor_event.record()
                        _pending_body_motor = _pinned_body_motor
                    elif tokenized.proprio is not None and _pending_body_motor is None:
                        body_schema.update(motor_out.detach().squeeze(0), tokenized.proprio)

                    # Apply developmental constraints + exploration noise for action writing
                    if action_writer is not None:
                        motor_for_action = apply_developmental_constraints(
                            motor_out.detach().clone(), dev_limits)
                        # DIRECT mode: no exploration noise (wellness gradient is sole signal)
                        if wellness_mode != WellnessMode.DIRECT:
                            noise_scale = dev_limits.get("exploration_noise", 0.0)
                            if noise_scale > 0:
                                noise = torch.randn_like(motor_for_action) * noise_scale
                                motor_for_action[..., :3] += noise[..., :3]
                                motor_for_action[..., -1] += noise[..., -1].abs()
                                motor_for_action[..., 8:9].clamp_(0.0, dev_limits["strength_limit"])
                        action_writer.write(
                            motor_commands=motor_for_action.squeeze(0),
                            gaze_target=gaze_out.squeeze(0) if gaze_out is not None else None,
                            voice_params=voice_out.squeeze(0) if voice_out is not None else None,
                        )

                    if use_timing:
                        evt_motor.record()

                    t_forward = time.perf_counter()

                    motor_target = handle_state[:, :, :9]

                    # Imagination chain (UEMA Spec §6.3): retroactive multi-step prediction.
                    # Uses compiled forward_model + prediction_heads for ~4x speedup
                    # (80ms → 15ms for 3 steps) vs the old _orig_mod path.
                    imagination_loss = None
                    imag_step_losses = []
                    if (imagination_enabled
                            and step % imagination_every_n == 0
                            and len(encoder_output_history) >= imagination_steps + 1
                            and len(motor_history) >= imagination_steps):
                        imagination_loss, imag_step_losses = _run_imagination_compiled(
                            encoder_output_history, motor_history, imagination_steps,
                            forward_model, prediction_heads, encoder_registry,
                            context_buffer, active, n_mod,
                            latent_regularizer=latent_regularizer,
                            latent_reg_weight=latent_reg_weight,
                            valid_history=list(valid_history),
                            achieved_state=handle_state,
                        )

                    # Mean PE (always computed — used by curiosity and/or curriculum).
                    # Synthetic (placeholder) modalities are excluded: their targets are
                    # noise, so their PE is always high and would pin the curiosity drive
                    # at "overwhelmed" (found on the MIMo test bed, 2026-08-19).
                    with torch.no_grad():
                        pe_vals = [F.mse_loss(predictions[m], targets[m]).item()
                                   for m in predictions if m in targets
                                   and _src_to_str(source_mask.get(m, "real")) != "synthetic"]
                        mean_pe = sum(pe_vals) / len(pe_vals) if pe_vals else 0.0

                    # Curiosity drive [0,1]: amplifies wellness loss in goldilocks PE zone
                    curiosity_drive = 1.0
                    if curiosity_system is not None:
                        curiosity_drive = curiosity_system.compute_drive(mean_pe)

                    # Apply curiosity drive as scalar on wellness loss terms
                    # [0,1]: 0 = no amplification (overwhelmed/bored), 1 = full signal (goldilocks peak)
                    scaled_total_wellness = (
                        curiosity_drive * total_wellness
                        if wellness_mode == WellnessMode.DIRECT and total_wellness is not None
                        else None
                    )
                    scaled_forward_wellness = (
                        curiosity_drive * forward_wellness_loss
                        if forward_wellness_loss is not None
                        else None
                    )

                    # Intention generator (CF-9): inference-only planning
                    intention_plan_meta = None
                    if (intention_enabled
                            and intention_generator is not None
                            and not forward_model.training
                            and len(encoder_output_history) >= 2):
                        from lem.model.intention_generator import plan_multistep as _plan_ms
                        with torch.no_grad():
                            _ig_current = encoder_output_history[-2]
                            _ig_goal = encoder_output_history[-1]
                            _ig_motor = (motor_history[-1]
                                         if motor_history else motor_out.detach())
                            _, intention_plan_meta = _plan_ms(
                                intention_generator, forward_model,
                                encoder_registry, prediction_heads,
                                wellness.score_predicted_wellness,
                                _ig_current, _ig_goal, _ig_motor,
                                max_steps=intention_cfg.get("max_plan_steps", 5))

                    # Latent regularizer penalty on single-step predictions
                    latent_reg_penalty = None
                    if latent_regularizer is not None:
                        latent_reg_penalty = latent_regularizer.penalty(predictions)

                    # When imagination backward is decoupled (async stream), exclude
                    # imagination_loss from the main loss so backward graphs are separate.
                    imag_loss_weighted = None
                    if imagination_stream is not None:
                        main_loss, loss_bd = loss_manager.compute(
                            predictions, targets, source_mask,
                            motor_out, motor_target, wellness_out,
                            dev_limits, step,
                            gaze_out=gaze_out, gaze_target=gaze_target,
                            motor_pos_excursion=motor_meta.get("pos_excursion"),
                            return_breakdown=True,
                            total_wellness=scaled_total_wellness,
                            forward_wellness_loss=scaled_forward_wellness,
                            imagination_loss=None,
                            latent_reg_loss=latent_reg_penalty,
                            encoder_outputs=tokens)
                        if imagination_loss is not None:
                            imag_weight = config["training"].get("imagination", {}).get("loss_weight", 0.5)
                            imag_loss_weighted = imag_weight * imagination_loss
                            if loss_bd is not None:
                                loss_bd["imagination"] = imag_loss_weighted.detach()
                        loss = (main_loss + imag_loss_weighted) if imag_loss_weighted is not None else main_loss
                    else:
                        main_loss = None
                        loss, loss_bd = loss_manager.compute(
                            predictions, targets, source_mask,
                            motor_out, motor_target, wellness_out,
                            dev_limits, step,
                            gaze_out=gaze_out, gaze_target=gaze_target,
                            motor_pos_excursion=motor_meta.get("pos_excursion"),
                            return_breakdown=True,
                            total_wellness=scaled_total_wellness,
                            forward_wellness_loss=scaled_forward_wellness,
                            imagination_loss=imagination_loss,
                            latent_reg_loss=latent_reg_penalty,
                            encoder_outputs=tokens)

                    # Accumulate loss breakdown (GPU-side)
                    if loss_bd is not None:
                        for mod_lb, val_lb in loss_bd.get("prediction", {}).items():
                            key_lb = f"pred_{mod_lb}"
                            if key_lb in loss_breakdown_accum:
                                loss_breakdown_accum[key_lb] += val_lb
                        for comp_lb in ("motor", "wellness", "gaze", "vicreg",
                                        "motor_mse", "motor_wellness",
                                        "imagination", "latent_reg",
                                        "cross_sim"):
                            if comp_lb in loss_bd:
                                loss_breakdown_accum[comp_lb] += loss_bd[comp_lb]

                t_loss = time.perf_counter()
                if use_timing:
                    evt_loss.record()

                # Gradient probes (report steps only, BEFORE backward frees the graph)
                _step_grad_probes = None
                if (wellness_mode != WellnessMode.BASELINE
                        and step > 0 and step % args.report_every == 0):
                    _step_grad_probes = {}
                    # Formula probes: d(formula_component)/d(motor_commands)
                    # motor_commands is the SHM leaf — measures direct formula path
                    for probe_name in ["prediction_error", "tactile_comfort",
                                       "capability", "coherence", "motor_state"]:
                        probe_tensor = wellness_out["grounded"].get(probe_name)
                        if probe_tensor is not None and probe_tensor.requires_grad:
                            try:
                                probe_grad = torch.autograd.grad(
                                    probe_tensor.sum(), motor_commands,
                                    retain_graph=True, allow_unused=True
                                )[0]
                                _step_grad_probes[probe_name] = (
                                    probe_grad.norm().item() if probe_grad is not None else 0.0
                                )
                            except RuntimeError:
                                _step_grad_probes[probe_name] = -1.0
                        else:
                            _step_grad_probes[probe_name] = 0.0

                    # Forward probe: d(forward_wellness_loss)/d(motor_out)
                    # motor_out is the motor controller output — measures the actual
                    # FORWARD mode gradient path through predict_from_action → approx evaluator
                    if (wellness_mode == WellnessMode.FORWARD
                            and forward_wellness_loss is not None):
                        try:
                            fwd_grad = torch.autograd.grad(
                                forward_wellness_loss, motor_out,
                                retain_graph=True, allow_unused=True
                            )[0]
                            _step_grad_probes["forward_approx"] = (
                                fwd_grad.norm().item() if fwd_grad is not None else 0.0
                            )
                        except RuntimeError:
                            _step_grad_probes["forward_approx"] = -1.0

                # Backward + optimizer step
                optimizer.zero_grad()

                if imagination_stream is not None:
                    # === DECOUPLED IMAGINATION BACKWARD ===
                    # Main backward runs SYNC on default stream (consumes forward
                    # graph activations before CUDAGraph can replay and overwrite).
                    # Imagination backward runs ASYNC on imagination_stream.
                    if use_scaler:
                        scaler.scale(main_loss).backward()
                    else:
                        main_loss.backward()

                    if step == 0:
                        validate_step(step, tokenized, tokens, predictions,
                                      motor_out, loss, all_params)

                    # Clone .grad out of CUDAGraph buffer pool. CUDAGraph replay
                    # (in perception frames) reuses the same buffer pool for forward
                    # AND backward, so .grad lives in graph buffers. Cloning moves
                    # them to regular memory so they survive CUDAGraph replays until
                    # optimizer.step() at the next training frame.
                    for p in all_params:
                        if p.grad is not None:
                            p.grad = p.grad.clone()

                    # Imagination backward: async on imagination_stream.
                    # Event ensures imag kernels start after main backward + clone.
                    if imag_loss_weighted is not None:
                        main_bwd_event = torch.cuda.Event()
                        main_bwd_event.record()
                        imagination_stream.wait_event(main_bwd_event)
                        with torch.cuda.stream(imagination_stream):
                            if use_scaler:
                                scaler.scale(imag_loss_weighted).backward()
                            else:
                                imag_loss_weighted.backward()
                        # Record completion event for GPU-side guard in perception frames.
                        imag_bwd_done_event = torch.cuda.Event()
                        imag_bwd_done_event.record(imagination_stream)
                        pending_imagination = True
                        deferred_loss = loss
                        deferred_grad_clip = grad_clip
                    else:
                        # No imagination this step — immediate optimizer step.
                        if use_scaler:
                            scaler.unscale_(optimizer)
                        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                            all_params, grad_clip, foreach=True, error_if_nonfinite=False)
                        grad_norm_accum += pre_clip_norm.item()
                        grad_norm_count += 1
                        if use_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()

                    if use_timing:
                        evt_bwd.record()

                elif backward_stream is not None:
                    # === EXISTING ASYNC BACKWARD (no decoupling) ===
                    with torch.cuda.stream(backward_stream):
                        if use_scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    deferred_loss = loss
                    pending_backward = True
                    if use_timing:
                        evt_bwd.record(backward_stream)

                else:
                    # === EXISTING SYNC BACKWARD ===
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if step == 0:
                        validate_step(step, tokenized, tokens, predictions,
                                      motor_out, loss, all_params)

                    if use_timing:
                        evt_bwd.record()

                    if use_scaler:
                        scaler.unscale_(optimizer)
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, grad_clip, foreach=True, error_if_nonfinite=False)
                    grad_norm_accum += pre_clip_norm.item()
                    grad_norm_count += 1
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()

                # Update context buffer: shift left by n_mod, append current frame
                if ctx_len > 1:
                    context_buffer = torch.cat([
                        context_buffer[:, n_mod:, :], ctx_update.detach()
                    ], dim=1)

                t_end = time.perf_counter()
                if use_timing:
                    evt_step.record()

                step += 1
                total_loss_accum += loss.detach()  # GPU-side accumulation, no sync
                frame_time_ms = (t_end - t0) * 1000
                frame_times.append(frame_time_ms)

                # Curriculum step (feed PE, check phase transition)
                if curriculum_mgr is not None:
                    cur_result = curriculum_mgr.step(mean_pe=mean_pe)
                    if cur_result["phase_changed"]:
                        new_limits = curriculum_mgr.get_dev_limits()
                        # Context buffer resize
                        new_ctx = new_limits["context_length"]
                        if new_ctx != ctx_len:
                            old_tokens = (ctx_len - 1) * n_mod
                            new_tokens = (new_ctx - 1) * n_mod
                            if new_tokens > old_tokens:
                                pad = torch.zeros(
                                    1, new_tokens - old_tokens, d_model, device=device)
                                context_buffer = torch.cat([pad, context_buffer], dim=1)
                            elif new_tokens < old_tokens:
                                context_buffer = context_buffer[:, -new_tokens:, :]
                            ctx_len = new_ctx
                            config["model"]["context_length"] = new_ctx
                        # Imagination toggle
                        imagination_enabled = new_limits["imagination_enabled"]
                        imagination_steps = new_limits["imagination_steps"]
                        # Reset loss accumulators (stale from old phase)
                        total_loss_accum = torch.zeros(1, device=device)
                        for k in loss_breakdown_accum:
                            loss_breakdown_accum[k].zero_()
                        grad_norm_accum = 0.0
                        grad_norm_count = 0
                        # Checkpoint at phase transition
                        checkpoint_mgr.save(
                            step, encoder_registry, forward_model, prediction_heads,
                            motor_controller, wellness, optimizer, receipt,
                            gaze_head=gaze_head, voice_head=voice_head, scaler=scaler,
                            body_schema=body_schema,
                            curriculum_state=curriculum_mgr.state_dict())
                        # Stats file entry
                        if stats_file is not None:
                            stats_file.write(json.dumps({
                                "type": "phase_transition",
                                "timestamp": time.time(),
                                **cur_result["transition_log"],
                            }) + "\n")
                            stats_file.flush()
                    if cur_result["progress_report"] is not None and stats_file is not None:
                        stats_file.write(json.dumps({
                            "type": "curriculum_progress",
                            "timestamp": time.time(),
                            "phase": curriculum_mgr.phase_name,
                            "phase_step": curriculum_mgr.phase_step,
                            "global_step": curriculum_mgr.global_step,
                        }) + "\n")
                        stats_file.flush()

                # CUDA Event timing — accumulate and report
                if use_timing:
                    torch.cuda.synchronize()
                    timing_accum["read"]  += evt_start.elapsed_time(evt_read)
                    timing_accum["xfer"]  += evt_read.elapsed_time(evt_xfer)
                    timing_accum["enc"]   += evt_xfer.elapsed_time(evt_encode)
                    timing_accum["fwd"]   += evt_encode.elapsed_time(evt_fwd)
                    timing_accum["motor"] += evt_fwd.elapsed_time(evt_motor)
                    timing_accum["loss"]  += evt_motor.elapsed_time(evt_loss)
                    timing_accum["bwd"]   += evt_loss.elapsed_time(evt_bwd)
                    timing_accum["optim"] += evt_bwd.elapsed_time(evt_step)
                    timing_accum["total"] += evt_start.elapsed_time(evt_step)
                    timing_count += 1

                # Periodic checkpoint save
                if args.save_every > 0 and step % args.save_every == 0:
                    checkpoint_mgr.save(
                        step, encoder_registry, forward_model, prediction_heads,
                        motor_controller, wellness, optimizer, receipt,
                        gaze_head=gaze_head, voice_head=voice_head, scaler=scaler,
                        body_schema=body_schema,
                        curriculum_state=(curriculum_mgr.state_dict()
                                          if curriculum_mgr is not None else None),
                    )
                    log.info(f"Checkpoint saved at step {step}")

                if step % args.report_every == 0:
                    avg_loss = total_loss_accum.item() / args.report_every  # single .item() on print step only
                    avg_time = sum(frame_times[-args.report_every:]) / min(len(frame_times), args.report_every)
                    read_ms = (t_read - t0) * 1000
                    encode_ms = (t_encode - t_read) * 1000
                    fwd_ms = (t_loss - t_encode) * 1000
                    bwd_ms = (t_end - t_loss) * 1000

                    perc_info = ""
                    if backward_every_n > 1 and perception_times:
                        avg_perc = sum(perception_times) / len(perception_times)
                        perc_info = f" perc_avg={avg_perc:.1f}ms({len(perception_times)}f)"
                        perception_times.clear()

                    log.info(
                        f"Step {step:4d} | loss={avg_loss:.4f} | "
                        f"frame={tokenized.metadata.frame_number} | "
                        f"phase={dev_limits['phase_name']} | "
                        f"train={frame_time_ms:.1f}ms "
                        f"(read={read_ms:.1f} enc={encode_ms:.1f} fwd={fwd_ms:.1f} bwd={bwd_ms:.1f}){perc_info} | "
                        f"sources={dict(source_mask)}"
                    )

                    # Gradient norm (pre-clip average over report interval)
                    if grad_norm_count > 0:
                        avg_grad_norm = grad_norm_accum / grad_norm_count
                        log.info(f"  Grad norm: avg={avg_grad_norm:.4f} clip={grad_clip:.1f}"
                                 f"{' (CLIPPED)' if avg_grad_norm > grad_clip else ''}")
                        grad_norm_accum = 0.0
                        grad_norm_count = 0

                    # Per-modality prediction error breakdown
                    pe_parts = []
                    for mod in predictions:
                        if mod in targets:
                            mod_pe = F.mse_loss(predictions[mod], targets[mod]).item()
                            pe_parts.append(f"{mod}={mod_pe:.4f}")
                    if pe_parts:
                        log.info(f"  PE breakdown: {' '.join(pe_parts)}")

                    if curiosity_system is not None:
                        log.info(f"  Curiosity: pe={mean_pe:.4f} drive={curiosity_drive:.4f} "
                                 f"zone={curiosity_system.zone(mean_pe)}")

                    if imagination_enabled:
                        if imag_step_losses:
                            log.info(f"  Imagination: loss={imagination_loss.item():.4f} "
                                     f"per_step=[{', '.join(f'{l.item():.4f}' for l in imag_step_losses)}]")
                        else:
                            log.info(f"  Imagination: warming up "
                                     f"({len(encoder_output_history)}/{imagination_steps + 1})")

                    # Motor output range (from clamped action output)
                    with torch.no_grad():
                        pos_vals = motor_out[..., :3]
                        str_vals = motor_out[..., 8:9]
                        log.info(
                            f"  Motor range: pos=[{pos_vals.min().item():.3f}, {pos_vals.max().item():.3f}] "
                            f"strength=[{str_vals.min().item():.3f}, {str_vals.max().item():.3f}] "
                            f"phase_limits=rom:{dev_limits['rom_limit']},str:{dev_limits['strength_limit']}"
                        )

                    # Vision encode ratio
                    log.info(
                        f"  Vision: {vision_encode_count}/{step} encoded, "
                        f"{step - vision_encode_count} reused"
                    )

                    # GPU device utilization with driver mode
                    if device.type == "cuda":
                        t_alloc = torch.cuda.memory_allocated(device) / 1024**2
                        t_resv = torch.cuda.memory_reserved(device) / 1024**2
                        t_props = torch.cuda.get_device_properties(device)
                        t_driver = "TCC" if t_props.is_integrated is False and getattr(t_props, 'tcc_driver', None) else "WDDM"
                        # Heuristic: on Windows dual-GPU, TCC GPU is enumerated first (cuda:0)
                        try:
                            import subprocess
                            smi = subprocess.run(
                                ["nvidia-smi", "--query-gpu=driver_model.current", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=2
                            )
                            if smi.returncode == 0:
                                drivers = [d.strip() for d in smi.stdout.strip().split('\n')]
                                # nvidia-smi orders by PCI bus; CUDA orders TCC before WDDM
                                tcc_indices = [i for i, d in enumerate(drivers) if d == "TCC"]
                                wddm_indices = [i for i, d in enumerate(drivers) if d == "WDDM"]
                                # cuda:0 = first TCC, cuda:1 = first WDDM (on this system)
                                cuda_driver_map = {}
                                cuda_idx = 0
                                for _ in tcc_indices:
                                    cuda_driver_map[cuda_idx] = "TCC"
                                    cuda_idx += 1
                                for _ in wddm_indices:
                                    cuda_driver_map[cuda_idx] = "WDDM"
                                    cuda_idx += 1
                                t_driver = cuda_driver_map.get(device.index or 0, "?")
                                c_driver = cuda_driver_map.get(capture_device.index or 0, "?") if multi_gpu else ""
                        except Exception:
                            t_driver = "?"
                            c_driver = "?"

                        gpu_line = f"  GPU | train({device} {t_driver}): {t_alloc:.0f}MB alloc / {t_resv:.0f}MB resv"
                        if multi_gpu and capture_device.type == "cuda":
                            c_alloc = torch.cuda.memory_allocated(capture_device) / 1024**2
                            c_resv = torch.cuda.memory_reserved(capture_device) / 1024**2
                            if 'c_driver' not in dir():
                                c_driver = "?"
                            gpu_line += f" | capture({capture_device} {c_driver}): {c_alloc:.0f}MB alloc / {c_resv:.0f}MB resv"
                        log.info(gpu_line)

                    # Stale modality tracking
                    stale_mods = [m for m in active if source_mask.get(m) not in ("real", "synthetic")]
                    if stale_mods:
                        log.info(f"  Stale modalities (using cached tokens): {stale_mods}")

                    # ── New stats (flush GPU accumulators) ──
                    re = args.report_every

                    # 1. Loss component breakdown
                    lb_parts = []
                    lb_dict = {}
                    for lb_k, lb_v in loss_breakdown_accum.items():
                        avg_v = lb_v.item() / re
                        lb_parts.append(f"{lb_k}={avg_v:.4f}")
                        lb_dict[lb_k] = avg_v
                        lb_v.zero_()
                    if lb_parts:
                        log.info(f"  Loss breakdown: {' '.join(lb_parts)}")

                    # 2. Per-encoder latency (wall-clock)
                    et_parts = []
                    et_dict = {}
                    for et_mod in active:
                        if encoder_timing_accum.get(et_mod, 0.0) > 0:
                            avg_et = encoder_timing_accum[et_mod] / re
                            et_parts.append(f"{et_mod}={avg_et:.1f}ms")
                            et_dict[et_mod] = avg_et
                            encoder_timing_accum[et_mod] = 0.0
                    if et_parts:
                        log.info(f"  Encoder latency: {' '.join(et_parts)}")

                    # 3. Motor confidence
                    conf_avg = confidence_sum.item() / re
                    conf_min_v = confidence_min.item()
                    conf_max_v = confidence_max.item()
                    log.info(f"  Motor confidence: avg={conf_avg:.3f} min={conf_min_v:.3f} max={conf_max_v:.3f}")
                    confidence_sum.zero_()
                    confidence_min.fill_(1.0)
                    confidence_max.zero_()

                    # 3b. Motor depth (CF-7) — averaged over ALL motor calls (train + perception)
                    depth_avg = motor_depth_accum / max(motor_depth_count, 1)
                    depth_min_v = motor_depth_min
                    depth_max_v = motor_depth_max
                    log.info(f"  Motor depth: avg={depth_avg:.1f} min={depth_min_v} max={depth_max_v} "
                             f"threshold={_to_float(motor_meta.get('depth_threshold', 0.0)):.3f}")
                    motor_depth_accum = 0
                    motor_depth_min = mc_cfg.get("max_depth", 6)
                    motor_depth_max = 0
                    motor_depth_count = 0

                    # 4. Handle utilization (top 3 by ROM and strength)
                    rom_avg = (handle_rom_accum / re).cpu().tolist()
                    str_avg = (handle_str_accum / re).cpu().tolist()
                    top3_rom = sorted(enumerate(rom_avg), key=lambda x: -x[1])[:3]
                    top3_str = sorted(enumerate(str_avg), key=lambda x: -x[1])[:3]
                    rom_str = " ".join(f"{HANDLE_NAMES[i]}={v:.3f}" for i, v in top3_rom)
                    str_str = " ".join(f"{HANDLE_NAMES[i]}={v:.3f}" for i, v in top3_str)
                    log.info(f"  Handle util (top3): ROM=[{rom_str}] STR=[{str_str}]")
                    handle_rom_dict = {HANDLE_NAMES[i]: rom_avg[i] for i in range(n_handles)}
                    handle_str_dict = {HANDLE_NAMES[i]: str_avg[i] for i in range(n_handles)}
                    handle_rom_accum.zero_()
                    handle_str_accum.zero_()

                    # 5. Gradient flow (per-component, computed only on report step)
                    component_params = {
                        "encoders": list(encoder_registry.parameters()),
                        "forward": list(forward_model.parameters()),
                        "pred_heads": list(prediction_heads.parameters()),
                        "motor_ctrl": list(motor_controller.parameters()),
                        "wellness": list(wellness.parameters()),
                    }
                    if gaze_head is not None:
                        component_params["gaze"] = list(gaze_head.parameters())
                    if voice_head is not None:
                        component_params["voice"] = list(voice_head.parameters())
                    gn_parts = []
                    gn_dict = {}
                    dead_components = []
                    for gn_name, gn_params in component_params.items():
                        gn_grads = [p.grad for p in gn_params if p.grad is not None]
                        if gn_grads:
                            gn_norm = torch.nn.utils.clip_grad_norm_(gn_params, float('inf')).item()
                            gn_parts.append(f"{gn_name}={gn_norm:.4f}")
                            gn_dict[gn_name] = gn_norm
                            if gn_norm < 1e-7:
                                dead_components.append(gn_name)
                        else:
                            dead_components.append(gn_name)
                    if gn_parts:
                        log.info(f"  Grad norms: {' '.join(gn_parts)}")
                    if dead_components:
                        log.warning(f"  DEAD LAYERS (zero grad): {dead_components}")

                    # 6. Encoder activation stats
                    ea_parts = []
                    ea_dict = {}
                    collapse_warnings = []
                    for ea_mod in active:
                        ea_mean = enc_act_mean_accum[ea_mod].item() / re
                        ea_std = enc_act_std_accum[ea_mod].item() / re
                        ea_parts.append(f"{ea_mod}(m={ea_mean:.3f},s={ea_std:.3f})")
                        ea_dict[ea_mod] = {"mean": ea_mean, "std": ea_std}
                        if ea_std < 1e-4:
                            collapse_warnings.append(ea_mod)
                        enc_act_mean_accum[ea_mod].zero_()
                        enc_act_std_accum[ea_mod].zero_()
                    if ea_parts:
                        log.info(f"  Encoder activations: {' '.join(ea_parts)}")
                    if collapse_warnings:
                        log.warning(f"  ENCODER COLLAPSE WARNING (std < 1e-4): {collapse_warnings}")

                    # 7. Wellness component values
                    w_parts = []
                    w_dict = {}
                    for wname in ["prediction_error", "tactile_comfort",
                                  "capability", "coherence", "motor_state"]:
                        avg_w = wellness_component_accum[wname].item() / re
                        w_parts.append(f"{wname}={avg_w:.4f}")
                        w_dict[wname] = avg_w
                        wellness_component_accum[wname].zero_()
                    avg_total_w = wellness_total_accum.item() / re
                    w_parts.append(f"total={avg_total_w:.4f}")
                    w_dict["total"] = avg_total_w
                    wellness_total_accum.zero_()
                    log.info(f"  Wellness: {' '.join(w_parts)}")

                    # 8. Motor output health
                    mh_mean = motor_health_mean_accum.item() / re
                    mh_std = motor_health_std_accum.item() / re
                    mh_min = motor_health_min_accum.item()
                    mh_max = motor_health_max_accum.item()
                    mh_div = motor_health_diversity_accum.item() / re
                    mh_dict = {"mean": mh_mean, "std": mh_std,
                               "min": mh_min, "max": mh_max, "diversity": mh_div}
                    log.info(f"  Motor health: mean={mh_mean:.4f} std={mh_std:.4f} "
                             f"min={mh_min:.4f} max={mh_max:.4f} diversity={mh_div:.4f}")
                    if motor_out is not None:
                        _mo = motor_out.detach().squeeze(0)  # [n_handles, 9]
                        log.info(f"  Motor commands: pos={_mo[:,:3].mean():.4f}±{_mo[:,:3].std():.4f} "
                                 f"radius={_mo[:,7].mean():.4f} strength={_mo[:,8].mean():.4f}")
                    motor_health_mean_accum.zero_()
                    motor_health_std_accum.zero_()
                    motor_health_min_accum.fill_(float('inf'))
                    motor_health_max_accum.fill_(float('-inf'))
                    motor_health_diversity_accum.zero_()

                    # 9. Per-component gradient probes (computed before backward, logged here)
                    wellness_grad_probes = _step_grad_probes or {}
                    if wellness_grad_probes:
                        # Formula probes: d(formula)/d(motor_commands) — direct path signal
                        formula_parts = [f"{k}={v:.6f}" for k, v in wellness_grad_probes.items()
                                         if k != "forward_approx"]
                        if formula_parts:
                            log.info(f"  Wellness grad probes: {' '.join(formula_parts)}")
                        # Forward probe: d(forward_wellness_loss)/d(motor_out) — forward path signal
                        if "forward_approx" in wellness_grad_probes:
                            log.info(f"  Forward wellness grad: {wellness_grad_probes['forward_approx']:.6f}")

                    # 10. Async reader ring utilization
                    ar_dict = None
                    if async_reader is not None:
                        ar_stats = async_reader.get_stats()
                        ar_captured = ar_stats["captured"] - last_async_stats["captured"]
                        ar_dropped = ar_stats["dropped"] - last_async_stats["dropped"]
                        ar_throttled = ar_stats["throttle_sleeps"] - last_async_stats["throttle_sleeps"]
                        ar_drop_pct = ar_dropped / max(ar_captured + ar_dropped, 1) * 100
                        log.info(f"  Async ring: fill={ar_stats['fill_level']}/{ar_stats['ring_size']} "
                                 f"drops={ar_dropped} ({ar_drop_pct:.0f}%) throttled={ar_throttled}")
                        ar_dict = {"fill_level": ar_stats["fill_level"],
                                   "drops": ar_dropped, "drop_rate_pct": ar_drop_pct,
                                   "throttled": ar_throttled}
                        last_async_stats = {"captured": ar_stats["captured"],
                                            "dropped": ar_stats["dropped"],
                                            "throttle_sleeps": ar_stats["throttle_sleeps"]}

                    # ── JSON Lines output ──
                    stats_record = {
                        "step": step,
                        "timestamp": time.time(),
                        "frame_number": tokenized.metadata.frame_number,
                        "phase": dev_limits["phase_name"],
                        "avg_loss": avg_loss,
                        "frame_time_ms": avg_time,
                        "loss_breakdown": lb_dict,
                        "encoder_latency_ms": et_dict,
                        "motor_confidence": {"avg": conf_avg, "min": conf_min_v, "max": conf_max_v},
                        "motor_depth": {"avg": depth_avg, "min": depth_min_v, "max": depth_max_v,
                                        "threshold": _to_float(motor_meta.get("depth_threshold", 0.0))},
                        "handle_utilization": {"rom": handle_rom_dict, "strength": handle_str_dict},
                        "grad_norms": gn_dict,
                        "encoder_activations": ea_dict,
                        "vision_encode_count": vision_encode_count,
                        "sources": dict(source_mask),
                        "wellness_mode": args.wellness_mode,
                        "motor_state_formula": args.motor_state_formula,
                        "wellness_components": w_dict,
                        "motor_health": mh_dict,
                    }
                    # Per-report motor command snapshot for verification
                    if motor_out is not None:
                        _mo = motor_out.detach().squeeze(0)  # [n_handles, 9]
                        stats_record["motor_commands"] = {
                            "pos_mean": round(_mo[:, :3].mean().item(), 4),
                            "pos_std": round(_mo[:, :3].std().item(), 4),
                            "rot_norm_mean": round(_mo[:, 3:7].norm(dim=-1).mean().item(), 4),
                            "radius_mean": round(_mo[:, 7].mean().item(), 4),
                            "strength_mean": round(_mo[:, 8].mean().item(), 4),
                            "handle_pos_norms": [round(v, 4) for v in _mo[:, :3].norm(dim=-1).tolist()],
                        }
                    if wellness_grad_probes:
                        stats_record["wellness_grad_probes"] = wellness_grad_probes
                    if curiosity_system is not None:
                        stats_record["curiosity"] = {
                            "pe": mean_pe,
                            "drive": curiosity_drive,
                            "zone": curiosity_system.zone(mean_pe),
                        }
                    if imagination_enabled:
                        if imag_step_losses:
                            stats_record["imagination"] = {
                                "loss": imagination_loss.item(),
                                "per_step": [l.item() for l in imag_step_losses],
                                "steps": imagination_steps,
                                "every_n": imagination_every_n,
                            }
                        else:
                            stats_record["imagination"] = {"warming_up": True}
                    if intention_enabled:
                        if intention_plan_meta is not None:
                            stats_record["intention"] = {
                                "plan_length": intention_plan_meta["plan_length"],
                                "converged": intention_plan_meta.get("converged", False),
                                "wellness_exit": intention_plan_meta.get("wellness_exit", False),
                                "final_mse": intention_plan_meta.get("final_mse"),
                            }
                        else:
                            stats_record["intention"] = {"status": "waiting"}
                    if ar_dict is not None:
                        stats_record["async_ring"] = ar_dict
                    stats_record["train_mean_pe"] = round(mean_pe, 6)
                    stats_record["coherence_score"] = round(
                        body_schema.coherence_score(), 4)
                    if device.type == "cuda":
                        stats_record["vram_mb"] = {
                            "allocated": torch.cuda.memory_allocated(device) / 1024**2,
                            "reserved": torch.cuda.memory_reserved(device) / 1024**2,
                            "total": _gpu_info["total_vram_mb"] if _gpu_info else 0,
                        }
                        stats_record["peak_vram_mb"] = round(
                            torch.cuda.max_memory_allocated(device) / 1024**2, 0)
                        stats_record["alloc_retries"] = torch.cuda.memory_stats(
                            device).get("num_alloc_retries", 0)
                    if _xfer_timing_count > 0:
                        stats_record["transfer_times_ms"] = {
                            k: round(v / _xfer_timing_count, 2)
                            for k, v in _xfer_timing_accum.items()
                        }
                        _xfer_timing_accum = {}
                        _xfer_timing_count = 0
                    if _gpu_info is not None and not _gpu_info_emitted:
                        stats_record["gpu_info"] = _gpu_info
                        _gpu_info_emitted = True
                    stats_file.write(json.dumps(stats_record) + "\n")
                    stats_file.flush()

                    # CUDA Event timing breakdown (GPU-accurate)
                    if use_timing and timing_count > 0:
                        n = timing_count
                        log.info(
                            f"  GPU TIMING (avg {n} steps) | "
                            f"read={timing_accum['read']/n:.1f}ms "
                            f"xfer={timing_accum['xfer']/n:.1f}ms "
                            f"enc={timing_accum['enc']/n:.1f}ms "
                            f"fwd={timing_accum['fwd']/n:.1f}ms "
                            f"motor={timing_accum['motor']/n:.1f}ms "
                            f"loss={timing_accum['loss']/n:.1f}ms "
                            f"bwd={timing_accum['bwd']/n:.1f}ms "
                            f"optim={timing_accum['optim']/n:.1f}ms "
                            f"total={timing_accum['total']/n:.1f}ms"
                        )
                        for k in timing_accum:
                            timing_accum[k] = 0.0
                        timing_count = 0

                    total_loss_accum.zero_()  # reset GPU accumulator

            else:
                # ============================================================
                # PERCEPTION-ONLY FRAME: forward with no_grad, motor output,
                # action write. No backward, no optimizer, no graph retained.
                # ============================================================
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                        tokens = encoder_registry.encode(frame, source_mask)

                        if source_mask.get("vision") == "stale" and last_vision_token is not None:
                            tokens["vision"] = last_vision_token
                        elif "vision" in tokens:
                            last_vision_token = tokens["vision"].detach()

                        # General stale token reuse (perception-only path)
                        for mod in active:
                            if mod == "vision":
                                continue
                            if source_mask.get(mod) == "real" and mod in tokens:
                                last_tokens[mod] = tokens[mod].detach()
                            elif source_mask.get(mod) != "real" and mod in last_tokens:
                                tokens[mod] = last_tokens[mod]

                        # Normalize token dict: fixed key set AND order (see training path)
                        _zero = torch.zeros(1, 1, d_model, device=device)
                        tokens = {m: tokens.get(m, _zero) for m in active}

                        # GPU-side sync for pipelined vision (see training path)
                        if vision_enc_device is not None:
                            encoder_registry_raw = getattr(
                                encoder_registry, '_orig_mod', encoder_registry)
                            encoder_registry_raw.sync_vision()

                        latents, last_step, ctx_update = forward_model(tokens, context_buffer)
                        motor_out, _perc_confidence, _perc_meta = motor_controller(latents, handle_state)

                        # Accumulate motor depth stats (CF-7) — perception-only path
                        _step_depth = _perc_meta.get("depth", mc_cfg["n_layers"])
                        motor_depth_accum += _step_depth
                        motor_depth_min = min(motor_depth_min, _step_depth)
                        motor_depth_max = max(motor_depth_max, _step_depth)
                        motor_depth_count += 1

                        gaze_out = None
                        if gaze_head is not None:
                            gaze_out = gaze_head(last_step)

                        voice_out = None
                        if voice_head is not None:
                            voice_out = voice_head(last_step)

                # Update context buffer (perception frames also contribute temporal history)
                if ctx_len > 1:
                    context_buffer = torch.cat([
                        context_buffer[:, n_mod:, :], ctx_update.detach()
                    ], dim=1)

                # Update body schema (perception frames still contribute to correlation tracking)
                if _pending_body_motor is not None and tokenized.proprio is not None:
                    body_schema.update(_pending_body_motor, tokenized.proprio)
                if _pinned_body_motor is not None:
                    _body_motor_event.synchronize()
                    _pinned_body_motor.copy_(motor_out.detach().squeeze(0), non_blocking=True)
                    _body_motor_event.record()
                    _pending_body_motor = _pinned_body_motor
                elif tokenized.proprio is not None and _pending_body_motor is None:
                    body_schema.update(motor_out.detach().squeeze(0), tokenized.proprio)

                # Apply developmental constraints + exploration noise for action writing
                if action_writer is not None:
                    motor_for_action = apply_developmental_constraints(
                        motor_out.detach().clone(), dev_limits)
                    noise_scale = dev_limits.get("exploration_noise", 0.0)
                    if noise_scale > 0:
                        noise = torch.randn_like(motor_for_action) * noise_scale
                        motor_for_action[..., :3] += noise[..., :3]
                        motor_for_action[..., -1] += noise[..., -1].abs()
                        motor_for_action[..., 8:9].clamp_(0.0, dev_limits["strength_limit"])
                    action_writer.write(
                        motor_commands=motor_for_action.squeeze(0),
                        gaze_target=gaze_out.squeeze(0) if gaze_out is not None else None,
                        voice_params=voice_out.squeeze(0) if voice_out is not None else None,
                    )

                t_end = time.perf_counter()
                perception_times.append((t_end - t0) * 1000)

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        # Resolve any in-flight async backward before saving checkpoint
        if pending_backward and backward_stream is not None:
            backward_stream.synchronize()
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                    all_params, deferred_grad_clip, foreach=True, error_if_nonfinite=False)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            deferred_loss = None
            pending_backward = False

        if pending_imagination and imagination_stream is not None:
            imagination_stream.synchronize()
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                    all_params, deferred_grad_clip, foreach=True, error_if_nonfinite=False)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            deferred_loss = None
            pending_imagination = False

        stats_file.close()
        # Save final checkpoint
        if step > start_step:
            log.info(f"Saving final checkpoint at step {step}...")
            checkpoint_mgr.save(
                step, encoder_registry, forward_model, prediction_heads,
                motor_controller, wellness, optimizer, receipt,
                gaze_head=gaze_head, voice_head=voice_head, scaler=scaler,
                body_schema=body_schema,
                curriculum_state=(curriculum_mgr.state_dict()
                                  if curriculum_mgr is not None else None),
            )
        if action_writer is not None:
            action_writer.close()
        # Close async reader (which internally closes tokenizer) or tokenizer directly
        if async_reader is not None:
            async_reader.close()
        else:
            tokenizer.close()

    # Summary
    if frame_times:
        warmup_n = min(args.warmup, len(frame_times) - 1)
        steady = frame_times[warmup_n:]
        if steady:
            avg_ms = sum(steady) / len(steady)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        else:
            avg_ms = sum(frame_times) / len(frame_times)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        log.info(f"Done. {step} training steps ({frame_counter} total frames), "
                 f"avg train={avg_ms:.1f}ms/step ({fps:.1f} FPS) "
                 f"[warmup={warmup_n} steps excluded]")
        if backward_every_n > 1:
            n_perc = frame_counter - step
            log.info(f"  Perception-only frames: {n_perc} "
                     f"(backward_every_n={backward_every_n})")
        log.info(f"Tokenizer: {tokenizer.frames_processed} frames, {tokenizer.frames_skipped} skipped")
        if async_reader is not None:
            log.info(f"Async reader: {async_reader.frames_captured} captured, "
                     f"{async_reader.frames_dropped} dropped")
        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            log.info(f"Peak VRAM ({device}): {peak_mb:.0f} MB")
            if multi_gpu and capture_device.type == "cuda":
                cap_peak = torch.cuda.max_memory_allocated(capture_device) / 1024**2
                log.info(f"Peak VRAM ({capture_device}): {cap_peak:.0f} MB")


if __name__ == "__main__":
    main()
