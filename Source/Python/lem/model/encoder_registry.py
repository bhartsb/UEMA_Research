# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
EncoderRegistry — nn.Module containing nn.ModuleDict of active encoders.

Built dynamically from resolved config. encode() takes a frame and source_mask,
returns {modality_name: [1, d_model] tensor} for all non-disabled modalities.

Supports cross-device vision encoding: when vision_device is set, the vision
encoder runs on a separate GPU (e.g. cuda:1) while all other encoders stay on
the primary device (cuda:0). Only the tiny [B, 1, d_model] vision token crosses
GPUs instead of the full [B, N, 768] patch tensor (~95 MB → 3 KB).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ENCODER_FACTORIES
from lem.source_mask import SOURCE_DISABLED, to_int


class EncoderRegistry(nn.Module):
    """
    Registry of active modality encoders.

    Built from resolved config. Only active modalities get encoders.
    encode() dispatches per-modality data to the correct encoder and
    returns a dict of {modality_name: [B, 1, d_model]} tensors.
    """

    def __init__(self, config: dict):
        super().__init__()

        self._active_modalities = config["_active_modalities"]
        self._d_model = config["model"]["d_model"]

        # Build encoder for each active modality
        encoders = {}
        for mod_name in self._active_modalities:
            enc_cfg = config["modalities"][mod_name]["encoder"]
            enc_type = enc_cfg["type"]

            if enc_type not in ENCODER_FACTORIES:
                raise ValueError(
                    f"Unknown encoder type '{enc_type}' for modality '{mod_name}'. "
                    f"Available: {list(ENCODER_FACTORIES.keys())}"
                )

            encoders[mod_name] = ENCODER_FACTORIES[enc_type](enc_cfg)

        self.encoders = nn.ModuleDict(encoders)

        # SHARED OUTPUT LAYERNORM, and why it is not affine.
        #
        # predictive_state.py documents the invariant that a real encoder token has
        # RMS 1.0000. A learnable gain and bias here breaks that by construction, and
        # it did: measured across H31's eight hourly checkpoints, ||output_norm.bias||
        # went 3.21 -> 22.02 while the gain barely moved, so the share of every
        # token's energy that is one CONSTANT vector - the same for all six
        # modalities and all 64 timesteps - climbed 10.9% -> 76.4% and was still
        # rising when the run ended. Three quarters of the signal carried no
        # information. The gradient beats weight decay on that tensor by 3-19x, so
        # decay was never going to hold it.
        #
        # Nothing is lost by removing the affine: all six encoders already end in
        # their own norm, so per-modality scale is still learnable. What goes is the
        # one shared degree of freedom that could drift for every modality at once.
        #
        # affine=True reproduces the old behaviour for anyone rerunning an old
        # experiment. Resuming a pre-existing checkpoint under the default drops its
        # saved gain and bias, which changes token scale - intentionally, since that
        # bias is the thing being removed.
        _ed = config.get("encoder_defaults", {})
        use_output_ln = _ed.get("output_layernorm", True)
        affine = bool(_ed.get("output_layernorm_affine", False))
        self.output_norm = (nn.LayerNorm(self._d_model, elementwise_affine=affine)
                            if use_output_ln else None)

        # Cross-device vision encoding state (set via move_vision_to)
        self._vision_device: Optional[torch.device] = None
        self._primary_device: Optional[torch.device] = None
        self._vision_event: Optional[torch.cuda.Event] = None

    @property
    def active_modalities(self) -> list[str]:
        return list(self._active_modalities)

    def to(self, *args, **kwargs):
        """Override to track primary device and preserve vision device placement."""
        result = super().to(*args, **kwargs)
        # Infer primary device from a parameter
        for p in self.parameters():
            self._primary_device = p.device
            break
        # If vision device was set, re-move vision encoder there
        if self._vision_device is not None and "vision" in self.encoders:
            self.encoders["vision"] = self.encoders["vision"].to(self._vision_device)
        return result

    def move_vision_to(self, vision_device: torch.device) -> None:
        """Move the vision encoder to a separate device for pipelined encoding.

        After this call, vision encoding runs on vision_device and only the
        tiny output token [B, 1, d_model] is transferred to the primary device.
        """
        if "vision" not in self.encoders:
            return
        self._vision_device = torch.device(vision_device)
        self.encoders["vision"] = self.encoders["vision"].to(self._vision_device)
        # Pre-allocate event for GPU-side synchronization
        if self._vision_device.type == "cuda":
            self._vision_event = torch.cuda.Event()

    def sync_vision(self) -> None:
        """Wait for cross-device vision encoding to complete.

        Uses GPU-side event wait: the default stream on primary_device waits
        for the vision_device stream to finish. Zero CPU overhead — only blocks
        the GPU command queue if vision encoding hasn't finished yet.
        """
        if self._vision_event is not None:
            torch.cuda.current_stream(self._primary_device).wait_event(self._vision_event)

    def _encode_one(self, mod_name: str, data, encoder: nn.Module) -> torch.Tensor:
        """Run a single encoder. Handles cross-device vision and data format dispatch."""
        cross_device_vision = (
            mod_name == "vision"
            and self._vision_device is not None
            and isinstance(data, dict)
        )

        if cross_device_vision:
            # Vision data is already on vision_device (routed by build_frame)
            out = encoder(data["patches"], data.get("angular_pos"))
            # Transfer tiny token [B, 1, d_model] to primary device
            out = out.to(self._primary_device, non_blocking=True)
            # Record event on vision_device so primary can GPU-wait
            if self._vision_event is not None:
                self._vision_event.record(torch.cuda.current_stream(self._vision_device))
            # output_norm runs on primary device (after transfer)
            if self.output_norm is not None:
                out = self.output_norm(out)
            return out

        # THE MOTOR WIDENING LIVES HERE, not in the frame.
        #
        # frame["motor"] is the PHYSICAL 10-field handle state, and five other consumers
        # read that layout: the motor controller, the motor loss target, the GPU replay
        # buffer, the shared-GPU ring and the wellness formulas. Widening the frame to 19
        # for the encoder's benefit broke every one of them - measured:
        #   controller   "mat1 and mat2 shapes cannot be multiplied (1x1440 and 1224x128)"
        #   replay store "Target sizes: [24, 10]. Tensor sizes: [24, 19]"
        # in live_test_lite, inference_frame, inference_process and the ring writer, none
        # of which the earlier narrowing patch touched.
        #
        # Doing it at the encoder boundary means the physical tensor stays physical
        # everywhere and only this encoder ever sees the command columns.
        if mod_name == "motor" and torch.is_tensor(data):
            want = getattr(getattr(encoder, "patch_mlp", [None])[0], "in_features", None)
            if want and data.shape[-1] < want:
                pad = torch.zeros(*data.shape[:-1], want - data.shape[-1],
                                  device=data.device, dtype=data.dtype)
                data = torch.cat([data, pad], dim=-1)

        # Standard path — all on primary device
        if mod_name == "vision" and isinstance(data, dict):
            out = encoder(data["patches"], data.get("angular_pos"))
        elif mod_name == "audio" and isinstance(data, dict):
            out = encoder(data["spectral"], data["spatial"])
        else:
            out = encoder(data)

        if self.output_norm is not None:
            out = self.output_norm(out)
        return out

    def encode(self, frame: dict,
               source_mask: Optional[Dict[str, str]] = None,
               ) -> Dict[str, torch.Tensor]:
        """
        Encode all active modalities from a frame.

        Args:
            frame: Dict of per-modality input tensors. Keys are modality names.
                   Vision: {"patches": [B, N, D], "angular_pos": [B, N, 2]}
                   Audio:  {"spectral": [B, F, C], "spatial": [B, S]}
                   Others: single tensor per modality
            source_mask: Optional dict of {modality: "real"|"synthetic"|"stale"|"disabled"}.
                         Modalities marked "disabled" are skipped.

        Returns:
            Dict of {modality_name: [B, 1, d_model]} for non-disabled modalities.
        """
        if source_mask is None:
            source_mask = {}

        tokens = {}
        for mod_name in self._active_modalities:
            if to_int(source_mask.get(mod_name, 0)) == SOURCE_DISABLED:
                continue
            if mod_name not in frame:
                continue
            tokens[mod_name] = self._encode_one(
                mod_name, frame[mod_name], self.encoders[mod_name])

        return tokens

    def encode_timed(self, frame: dict,
                     source_mask: Optional[Dict[str, str]] = None,
                     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Like encode(), but also returns wall-clock timing per encoder.

        Returns:
            (tokens, timings) where timings = {mod_name: milliseconds}.
            Timings measure CPU dispatch time (no GPU sync).
        """
        if source_mask is None:
            source_mask = {}

        tokens = {}
        timings = {}
        for mod_name in self._active_modalities:
            if to_int(source_mask.get(mod_name, 0)) == SOURCE_DISABLED:
                continue
            if mod_name not in frame:
                continue

            t0 = time.perf_counter()
            tokens[mod_name] = self._encode_one(
                mod_name, frame[mod_name], self.encoders[mod_name])
            timings[mod_name] = (time.perf_counter() - t0) * 1000.0

        return tokens, timings


COMMAND_POSITION_RANGE_CM = 200.0   # tokenizer.config.TokenizerConfig.position_range_cm


def normalize_executed_command(cmd, position_range_cm=COMMAND_POSITION_RANGE_CM):
    """An EXECUTED order (position in centimetres) in the units of the achieved half.

    The tokenizer divides every handle position on the wire by position_range_cm, so
    the achieved half of a motor row holds pos/200. An executed order - the output of
    apply_developmental_constraints / execution_transform, the bridge ledger, a planner
    candidate after the execution transform - holds raw centimetres. Concatenating the
    two unscaled put 99.6% of the row's input energy into the centimetre block and left
    rotation at 0.1% and strength at 0.0%, so the motor token encoded where the handles
    were rather than what they were told to do (measured on H33, 2026-08-27).

    Only the three position columns change; rotation, radius and strength are already
    dimensionless. A clone is returned; the caller's tensor (and the ledger) keep cm.
    Differentiable, so the planner's wellness gradient passes through it.

    NOT for the controller's raw output: motor_out positions are already normalized.
    The ledger (replay_train.last_outgoing_cmd) stores command CENTIMETRES and only
    commands a body actually received (2026-09-08 unit/validity contract; the old
    no-writer fallback that stored raw normalized proposals is gone) - so ledger
    rows DO pass through here, on every consumer path, replay and imagination alike.
    """
    out = cmd.clone()
    out[..., 0:3] = out[..., 0:3] / float(position_range_cm)
    return out


def build_motor_input(registry, action9, achieved=None):
    """Lay out the motor encoder's input the SAME way everywhere.

    The token is [achieved handle state (10), executed command (9)] once the transition
    contract is in force, and [command (9), is_active (1)] on the legacy 10-wide encoder.
    Every producer - the live step, the imagination chain, the planner - must use this
    function, because a token that means one thing when awake and another when imagining
    is what let the planner's wellness scorer be trained in one domain and used in
    another.

    achieved: [B, n_handles, 10] the last real handle state. None fills zeros, which is
    off-distribution; the planner paths do this today and it is one of the reasons the
    canonical-state work exists. Callers that have the real value must pass it.
    """
    import torch
    reg = getattr(registry, "_orig_mod", registry)
    enc = reg.encoders["motor"]
    first = getattr(enc, "patch_mlp", None)
    in_f = first[0].in_features if first is not None else action9.shape[-1] + 1
    B, N, F = action9.shape
    if in_f == F + 1:                                   # legacy: command + is_active
        ones = torch.ones(B, N, 1, device=action9.device, dtype=action9.dtype)
        return torch.cat([action9, ones], dim=-1)
    pad = in_f - F
    if achieved is None:
        ach = torch.zeros(B, N, pad, device=action9.device, dtype=action9.dtype)
    else:
        ach = achieved[..., :pad].to(action9.dtype)
    return torch.cat([ach, action9], dim=-1)


def imagined_motor_token(registry, motor_input):
    """The motor token of an IMAGINED row: a planner candidate, a step of the imagination chain, a probe's roll-out
    (transition(), predict_from_action, imagine_step, LemLite._latent_for_order). Encoded through the registry as
    every motor token is, so it gets the output LayerNorm every real token gets.

    With --motor-asked-change on (lem/model/motor_asked_change.py, 2026-10-03) the asked-change path beside the
    motor encoder runs here with its weights detached: an imagined row's pose is the pose the roll-out started
    from, 1 and 2 model steps old at depths 2 and 3 of the imagination chain, so its asked change is not the change
    the servo acted on, and the chain's loss carries no per-sense normalisation. The body copy of Oct 1 learned from
    that route and its touch prediction got worse on lived model steps (docs/gain_of_function_v4/agents/
    planner_repair_2026-09-30.md, sections 11 and 12); its repair was the rule used here: the weights learn only
    where they are judged, on rows whose pose and command belong to the same model step. The row keeps its gradient,
    so the planner's descent still passes through the path. Without the path this is registry.encode, unchanged."""
    reg = getattr(registry, "_orig_mod", registry)
    path = reg.encoders["motor"].__dict__.get("asked_change_path")
    if path is None:
        return reg.encode({"motor": motor_input}, {"motor": "real"})["motor"]
    with path.frozen():
        return reg.encode({"motor": motor_input}, {"motor": "real"})["motor"]
