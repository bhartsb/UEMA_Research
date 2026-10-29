# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
GPUReplayBuffer — GPU-resident ring buffer for experience replay training.

Stores raw sensor inputs (pre-encoder) on the training device. Vision stores
post-Swin backbone features (pool + proj are re-run with gradients during
training). All other modalities store their raw tokenizer output and are
re-encoded through the full encoder during training.

Supports batched sampling with context window retrieval and consecutiveness
validation for ring wraparound safety.
"""

from __future__ import annotations

import logging
import random
import statistics
from typing import Callable, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Sum-Tree for prioritized replay
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """Array-backed binary sum-tree. Each leaf holds a priority value;
    internal nodes hold the sum of their children. O(log N) sample and update.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self._tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        """Set priority for leaf and propagate up."""
        tree_idx = leaf_idx + self._capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta

    def sample(self, value: float) -> int:
        """Traverse tree to find the leaf whose cumulative range contains *value*."""
        idx = 0
        while idx < self._capacity - 1:  # while not a leaf
            left = 2 * idx + 1
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        return idx - (self._capacity - 1)  # convert to leaf index

    def get(self, leaf_idx: int) -> float:
        return float(self._tree[leaf_idx + self._capacity - 1])


# Modality presence bit flags (matches pipeline EModality ordering)
MOD_VISION = 1 << 0
MOD_AUDIO = 1 << 1
MOD_PROPRIOCEPTION = 1 << 2
MOD_MOTOR = 1 << 3
MOD_TACTILE = 1 << 4
MOD_VOICE = 1 << 5
MOD_INTERMIND = 1 << 6


# ══════════════════════════════════════════════════════════════════════════════
# Modality descriptor table for simple (single-buffer) modalities
# ══════════════════════════════════════════════════════════════════════════════

class _ModalityDescriptor:
    """Describes a simple modality with one GPU buffer and uniform store/gather."""
    __slots__ = ("name", "active_key", "frame_key", "result_key",
                 "buf_attr", "enabled_attr", "presence_flag", "shape_fn", "dtype")

    def __init__(self, *, name: str, active_key: str, frame_key: str,
                 result_key: str, buf_attr: str, enabled_attr: str,
                 presence_flag: int, shape_fn: Callable[[dict], Tuple[int, ...]],
                 dtype: torch.dtype = torch.float32):
        self.name = name
        self.active_key = active_key
        self.frame_key = frame_key
        self.result_key = result_key
        self.buf_attr = buf_attr
        self.enabled_attr = enabled_attr
        self.presence_flag = presence_flag
        self.shape_fn = shape_fn
        self.dtype = dtype


_SIMPLE_MODALITIES = [
    _ModalityDescriptor(
        name="proprio", active_key="proprioception", frame_key="proprioception",
        result_key="proprioception", buf_attr="_proprio_buf",
        enabled_attr="_proprio_enabled", presence_flag=MOD_PROPRIOCEPTION,
        shape_fn=lambda cfg: (cfg["modalities"]["proprioception"]["encoder"].get(
            "_input_dim", 276),),
    ),
    _ModalityDescriptor(
        name="motor", active_key="motor", frame_key="motor",
        result_key="motor", buf_attr="_motor_buf",
        enabled_attr="_motor_enabled", presence_flag=MOD_MOTOR,
        shape_fn=lambda cfg: (cfg["body"]["n_handles"],
                              cfg["modalities"]["motor"]["pipeline"].get(
                                  "features_per_handle", 10)),
    ),
    # THE EXECUTED ACTION. The 'motor' field above is the PHYSICAL handle state - the
    # pose the body reached. The order that carried it there is a different quantity and
    # needs its own storage: without it, encode_batch_from_buffer has nothing to put in
    # the command columns and fills them with zeros, so full-path replay is state-only
    # prediction wearing the name 'action-conditioned'. The LemLite ring has carried this
    # since 2026-08-25; the full buffer did not.
    #
    # 9 features: pos(3) + rot(4) + radius(1) + strength(1). No is_active - that belongs
    # to the physical record, not to the command.
    _ModalityDescriptor(
        name="executed_action", active_key="motor", frame_key="executed_action",
        result_key="executed_action", buf_attr="_executed_action_buf",
        enabled_attr="_executed_action_enabled", presence_flag=MOD_MOTOR,
        shape_fn=lambda cfg: (cfg["body"]["n_handles"], 9),
    ),
    _ModalityDescriptor(
        name="tactile", active_key="tactile", frame_key="tactile",
        result_key="tactile", buf_attr="_tactile_buf",
        enabled_attr="_tactile_enabled", presence_flag=MOD_TACTILE,
        shape_fn=lambda cfg: (256, 10),
    ),
    _ModalityDescriptor(
        name="voice", active_key="voice", frame_key="voice",
        result_key="voice", buf_attr="_voice_buf",
        enabled_attr="_voice_enabled", presence_flag=MOD_VOICE,
        shape_fn=lambda cfg: (cfg["modalities"]["voice"]["pipeline"].get(
            "input_dim", 56),) if "voice" in cfg["modalities"] else (56,),
    ),
]


class GPUReplayBuffer:
    """GPU-resident ring buffer for experience replay training.

    Pre-allocates contiguous GPU tensors for all modalities at init.
    Supports O(1) store and batched gather by index.
    """

    def __init__(self, capacity: int, config: dict, device: torch.device,
                 prioritized: bool = False):
        """
        Args:
            capacity: Number of transition slots.
            config: Resolved master config (from load_and_resolve).
            device: Training device (buffer lives here, typically cuda:0).
            prioritized: Enable sum-tree prioritized sampling.
        """
        self._capacity = capacity
        self._device = device
        self._active_modalities = config["_active_modalities"]
        self._context_length = config["model"]["context_length"]
        # True where the slot's OUTGOING action is known. The newest moment never
        # has one - it is back-filled when the next moment arrives - so a consumer
        # must check this rather than assume every slot is labelled.
        self.has_executed = torch.zeros(capacity, dtype=torch.bool, device=device)

        # ── Vision backbone features ──
        vis_enc = config["modalities"]["vision"]["encoder"]
        self._max_patches = vis_enc["_n_patches"]
        self._d_internal = vis_enc["_d_internal"]
        self._vision_enabled = "vision" in self._active_modalities

        if self._vision_enabled:
            self._vision_buf = torch.zeros(
                capacity, self._max_patches, self._d_internal,
                device=device, dtype=torch.float16)
            self._vision_n_patches = torch.zeros(
                capacity, dtype=torch.int64, device="cpu")
        else:
            self._vision_buf = None
            self._vision_n_patches = None

        # ── Audio ──
        self._audio_enabled = "audio" in self._active_modalities
        if self._audio_enabled:
            self._audio_spectral_buf = torch.zeros(
                capacity, 256, 2, device=device)
            self._audio_spatial_buf = torch.zeros(
                capacity, 2, device=device)
        else:
            self._audio_spectral_buf = None
            self._audio_spatial_buf = None

        # ── Simple modalities (proprio, motor, tactile, voice) ──
        for desc in _SIMPLE_MODALITIES:
            enabled = desc.active_key in self._active_modalities
            setattr(self, desc.enabled_attr, enabled)
            if enabled:
                shape = desc.shape_fn(config)
                setattr(self, desc.buf_attr, torch.zeros(
                    capacity, *shape, device=device, dtype=desc.dtype))
            else:
                setattr(self, desc.buf_attr, None)

        # Dimension aliases used elsewhere in the codebase
        self._proprio_dim = config["modalities"]["proprioception"]["encoder"].get(
            "_input_dim", 276) if self._proprio_enabled else 276
        self._n_handles = config["body"]["n_handles"]
        self._features_per_handle = config["modalities"]["motor"]["pipeline"].get(
            "features_per_handle", 10)

        # ── Motor capability (for wellness evaluator) ──
        n_bodysets = config["body"].get("n_bodysets", 10)
        # capability vector: capacity per body set + accum energy + total
        cap_dim = n_bodysets * 2 + 1
        self._motor_capability_buf = torch.zeros(
            capacity, cap_dim, device=device)
        self._cap_dim = cap_dim

        # ── Gaze target ──
        self._gaze_buf = torch.zeros(capacity, 2, device=device)

        # ── Cached encoded tokens (from inference, avoids re-encoding) ──
        # Per-modality [capacity, 1, d_model] — inference writes these alongside
        # raw data. Training reads them for no_grad paths (target, imagination,
        # context vision) instead of re-running the encoder.
        self._d_model = config["model"]["d_model"]
        self._token_bufs: dict[str, torch.Tensor] = {}
        for mod in self._active_modalities:
            self._token_bufs[mod] = torch.zeros(
                capacity, 1, self._d_model, device=device)
        self._token_valid = torch.zeros(capacity, dtype=torch.bool, device="cpu")

        # ── Metadata (CPU-side, lightweight) ──
        self._source_masks: list[Optional[dict]] = [None] * capacity
        self._frame_numbers = torch.zeros(capacity, dtype=torch.int64)
        self._sequence_ids = torch.zeros(capacity, dtype=torch.int64)
        self._valid = np.zeros(capacity, dtype=np.bool_)
        # WHICH RUN OF UNBROKEN WORLD CONTINUITY EACH SLOT BELONGS TO. sequence_ids is a
        # monotonic write counter that keeps incrementing straight through a world reset,
        # so on its own it says a pre-reset and a post-reset moment are consecutive. This
        # column is what mark_reset() bumps; store() refuses to back-fill an action across
        # it and sample_with_context() refuses a window that straddles it.
        self._episode_ids = np.zeros(capacity, dtype=np.int64)
        self._episode_id = 0

        # ── Modality presence bitfield per slot ──
        self._modality_present = torch.zeros(capacity, dtype=torch.int32)

        # ── Ring state ──
        # Plain ints by default; bind_shared_ring_state() replaces with
        # tensor-backed accessors for GIL-free C++ loop sharing.
        self._write_head = 0
        self._size = 0
        self._total_written = 0
        self._shared_ring_tensors = None  # set by bind_shared_ring_state()

        # ── Prioritized replay ──
        self._prioritized = prioritized
        if prioritized:
            self._sum_tree = SumTree(capacity)
            self._max_priority = 1.0
        else:
            self._sum_tree = None
            self._max_priority = 1.0

        # ── C++ native ops (loaded lazily for GIL reduction) ──
        self._all_modalities_enabled = (
            self._vision_enabled and self._audio_enabled
            and all(getattr(self, d.enabled_attr) for d in _SIMPLE_MODALITIES))
        try:
            from lem.native_ops import load_native_ops
            self._native_ops = load_native_ops()
        except Exception:
            self._native_ops = None

        # ── VRAM estimation check ──
        if device.type == "cuda":
            estimated_mb = self.estimate_vram_mb()
            props = torch.cuda.get_device_properties(device)
            total_vram_mb = props.total_memory / (1024 * 1024)
            if estimated_mb > total_vram_mb * 0.85:
                logger.warning(
                    "Buffer VRAM estimate %.0f MB exceeds 85%% of available "
                    "%.0f MB on %s. Consider reducing --buffer-size.",
                    estimated_mb, total_vram_mb, device)

    # ──────────────────────────────────────────────────────────────────────
    # Shared ring state (for GIL-free C++ loop)
    # ──────────────────────────────────────────────────────────────────────

    def bind_shared_ring_state(self):
        """Replace scalar ring state with CPU int64 tensors.

        Returns (write_head, size, total_written) tensors that can be passed
        to the C++ inference loop. Both Python properties and C++ code will
        read/write the same underlying memory.
        """
        wh = torch.tensor([self._write_head], dtype=torch.int64)
        sz = torch.tensor([self._size], dtype=torch.int64)
        tw = torch.tensor([self._total_written], dtype=torch.int64)
        self._shared_ring_tensors = (wh, sz, tw)
        return wh, sz, tw

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        if self._shared_ring_tensors is not None:
            return int(self._shared_ring_tensors[1].item())
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def fill_ratio(self) -> float:
        return self.size / self._capacity if self._capacity > 0 else 0.0

    @property
    def total_written(self) -> int:
        if self._shared_ring_tensors is not None:
            return int(self._shared_ring_tensors[2].item())
        return self._total_written

    # ──────────────────────────────────────────────────────────────────────
    # Episode boundaries
    # ──────────────────────────────────────────────────────────────────────

    def mark_reset(self) -> None:
        """The world discontinued: a teleport, a respawn, a new session, a rebuild that
        replaces the body.

        Slots written after this call may not share a training window with slots written
        before it, and no executed action may be back-filled across it. Same contract as
        mimo_bridge/replay.py ReplayMemory.mark_reset, which the bridge ring has carried
        since the cross-episode back-fill was removed there.

        NOTHING ON THE UE PATH CALLS THIS YET. lem/replay_train.py has no world-reset
        signal to wire it to, and lem/phase_rebuild.py hands this same buffer object to
        the new AsyncTrainer without calling it. Until a caller exists, the guards below
        are inert and the buffer behaves as one unbroken episode.
        """
        self._episode_id += 1

    # ──────────────────────────────────────────────────────────────────────
    # Store
    # ──────────────────────────────────────────────────────────────────────

    def store(self,
              vision_backbone_features: Optional[torch.Tensor],
              frame: dict,
              source_mask: dict,
              motor_capability: Optional[torch.Tensor],
              gaze_az_rad: float,
              gaze_el_rad: float,
              frame_number: int,
              executed_action: Optional[torch.Tensor] = None) -> int:
        """Store a single frame's raw inputs at the next ring slot.

        Args:
            vision_backbone_features: [N, d_internal] or [1, N, d_internal]
                post-Swin features on training device. None if stale.
            frame: Dict of raw sensor inputs (build_frame_from_tokenized format).
            source_mask: {mod: "real"|"stale"|"synthetic"} per modality.
            motor_capability: [cap_dim] or [1, cap_dim] wellness capability tensor.
            gaze_az_rad: Ground truth gaze azimuth in radians.
            gaze_el_rad: Ground truth gaze elevation in radians.
            frame_number: Pipeline frame number (int64).

        Returns:
            Slot index where data was stored.
        """
        slot = self._write_head
        presence = 0

        # BACK-FILL THE OUTGOING ACTION.
        #
        # The action that carries a moment forward does not exist when that moment is
        # stored - the model has not chosen it yet. It arrives with the NEXT store call,
        # which is why it is written one slot back rather than here. Storing the arriving
        # action at THIS slot would label every transition with the order that had
        # already finished acting, which is the defect the whole contract exists to
        # remove. Same rule as the LemLite ring.
        #
        # Never across an episode boundary: an action cannot carry a body through a
        # teleport, and labelling the pre-reset moment with the post-reset action would
        # manufacture exactly the false cause being removed. The previous slot's episode
        # id is what enforces that; mark_reset() bumps it. Before the id existed this
        # comment stood above a bare `if self._size > 0` and exactly one slot per reset
        # was back-filled with an order issued to the other body.
        if executed_action is not None and getattr(self, "_executed_action_enabled", False):
            prev = (slot - 1) % self._capacity
            if self._size > 0 and self._episode_ids[prev] == self._episode_id:
                a = executed_action
                if a.dim() == 3:
                    a = a[0]
                self._executed_action_buf[prev] = a.to(self._executed_action_buf.dtype)
                self.has_executed[prev] = True
            self.has_executed[slot] = False        # this moment's action is unknown yet

        # Vision
        if self._vision_enabled and vision_backbone_features is not None:
            vf = vision_backbone_features
            if vf.dim() == 3:
                vf = vf.squeeze(0)  # [1, N, D] -> [N, D]
            if vf.dtype != torch.float16:
                vf = vf.half()
            n_patches = vf.shape[0]
            self._vision_buf[slot, :n_patches, :] = vf
            if n_patches < self._max_patches:
                self._vision_buf[slot, n_patches:, :] = 0
            self._vision_n_patches[slot] = n_patches
            presence |= MOD_VISION

        # Audio
        if self._audio_enabled and "audio" in frame:
            audio = frame["audio"]
            if isinstance(audio, dict):
                spec = audio["spectral"]
                if spec.dim() == 3:
                    spec = spec.squeeze(0)
                self._audio_spectral_buf[slot] = spec
                spatial = audio.get("spatial")
                if spatial is not None:
                    if spatial.dim() == 2:
                        spatial = spatial.squeeze(0)
                    self._audio_spatial_buf[slot] = spatial
            presence |= MOD_AUDIO

        # Simple modalities (proprio, motor, tactile, voice)
        for desc in _SIMPLE_MODALITIES:
            if getattr(self, desc.enabled_attr) and desc.frame_key in frame:
                t = frame[desc.frame_key]
                buf = getattr(self, desc.buf_attr)
                if t.dim() >= buf.dim():  # has batch dim → squeeze
                    t = t.squeeze(0)
                buf[slot] = t
                presence |= desc.presence_flag

        # Motor capability
        if motor_capability is not None:
            mc = motor_capability
            if mc.dim() == 2:
                mc = mc.squeeze(0)
            n = min(mc.shape[0], self._cap_dim)
            self._motor_capability_buf[slot, :n] = mc[:n]
            if n < self._cap_dim:
                self._motor_capability_buf[slot, n:] = 0

        # Gaze
        self._gaze_buf[slot, 0] = gaze_az_rad
        self._gaze_buf[slot, 1] = gaze_el_rad

        # Metadata
        self._source_masks[slot] = dict(source_mask) if source_mask else {}
        self._frame_numbers[slot] = frame_number
        self._modality_present[slot] = presence
        self._valid[slot] = True
        self._episode_ids[slot] = self._episode_id
        self._total_written += 1
        self._sequence_ids[slot] = self._total_written

        # Priority (new slots get max_priority to ensure at least one sample)
        if self._sum_tree is not None:
            self._sum_tree.update(slot, self._max_priority)

        # Advance ring
        self._write_head = (self._write_head + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

        return slot

    def store_cached_tokens(self, slot: int,
                            tokens: dict[str, torch.Tensor]) -> None:
        """Store inference-computed encoded tokens for a ring slot.

        Called by the ingestion thread after copying raw data from the ring.
        Tokens are [1, d_model] per modality — the encoder output that
        inference already computed and would otherwise be thrown away.
        """
        for mod, tok in tokens.items():
            if mod in self._token_bufs:
                t = tok
                if t.dim() == 3:
                    t = t.squeeze(0)  # [1, 1, D] → [1, D]
                self._token_bufs[mod][slot] = t
        self._token_valid[slot] = True

    def gather_cached_tokens(self, indices: list[int],
                             modalities: list[str],
                             indices_tensor: torch.Tensor | None = None,
                             ) -> dict[str, torch.Tensor]:
        """Gather cached encoded tokens for a batch of indices.

        Returns {mod: [B, 1, d_model]} dict of pre-encoded tokens.
        Falls back to None for modalities without cached data.
        Used by training for no_grad paths (target, imagination, context vision).

        If *indices_tensor* is provided (a pre-built long tensor on the
        target device), the Python-list → tensor conversion is skipped.
        """
        idx = indices_tensor if indices_tensor is not None else torch.tensor(indices, dtype=torch.long)
        result = {}
        for mod in modalities:
            if mod in self._token_bufs:
                result[mod] = self._token_bufs[mod][idx]  # [B, 1, d_model]
        return result

    @property
    def has_cached_tokens(self) -> bool:
        """True if any stored frames have cached tokens."""
        return bool(self._token_valid.any())

    # ──────────────────────────────────────────────────────────────────────
    # Sample
    # ──────────────────────────────────────────────────────────────────────

    def sample_with_context(self,
                            batch_size: int,
                            context_length: int,
                            imagination_steps: int = 0,
                            prioritized: bool = False,
                            beta: float = 1.0,
                            ) -> Optional[tuple[list[int], list[list[int]], Optional[torch.Tensor]]]:
        """Sample batch_size valid indices with context windows.

        A sample at index i is valid if:
          - Slots [i-ctx+1, ..., i] are all valid with consecutive sequence_ids.
          - Slot i+1 exists and is consecutive (target for prediction).
          - Slots [i+1, ..., i+imag_steps] exist and are consecutive (imagination).
          - Every slot of that window carries the same episode id (mark_reset).

        The episode test lives here, not in the C++ gather: _gather_batch_native returns
        rows for indices this function already chose. A future C++ candidate builder has
        to apply the same test or it reintroduces windows that straddle a world reset.

        Args:
            batch_size: Number of samples to draw.
            context_length: Total context window (including current frame).
            imagination_steps: Additional successor frames needed for imagination.
            prioritized: Use sum-tree prioritized sampling (requires init with prioritized=True).
            beta: Importance sampling exponent (anneals from ~0.4 to 1.0).

        Returns:
            (primary_indices, context_windows, is_weights) or None if insufficient data.
            primary_indices: list of B ring slot indices for current frames.
            context_windows: list of B lists, each with (ctx-1) predecessor slot indices.
            is_weights: [B] tensor of importance sampling weights (None if not prioritized).
        """
        ctx_before = context_length - 1  # Predecessor frames needed
        needed_after = imagination_steps + 1  # Successors (target + imagination)
        window_size = ctx_before + 1 + needed_after

        # Use properties (not raw ints) — when C++ native loop is active,
        # the shared tensors are updated directly by C++ via data_ptr().
        # The raw Python int attributes (_size, _total_written) stay frozen.
        cur_size = self.size
        cur_total = self.total_written

        if cur_size < window_size:
            return None

        # Build candidate set — vectorized (replaces per-candidate Python loop)
        oldest_count = cur_total - cur_size + 1  # inclusive
        newest_count = cur_total  # inclusive
        min_count = oldest_count + ctx_before
        max_count = newest_count - needed_after

        if max_count < min_count:
            return None

        # Vectorized: check all candidates at once using numpy
        counts = np.arange(min_count, max_count + 1, dtype=np.int64)
        # Window offsets: [-ctx_before, ..., 0 (center), ..., +needed_after]
        offsets = np.arange(-ctx_before, needed_after + 1, dtype=np.int64)
        # All counts in each window: [N_candidates, window_size]
        all_counts = counts[:, None] + offsets[None, :]
        # Map to ring slots (count is 1-based)
        all_slots = (all_counts - 1) % self._capacity
        # Check sequence_ids match expected (consecutive writes)
        seq_ids = self._sequence_ids.numpy()
        actual_seq = seq_ids[all_slots]
        seq_ok = (actual_seq == all_counts).all(axis=1)
        # Check all slots in window are valid
        valid_ok = self._valid[all_slots].all(axis=1)
        # Consecutive sequence ids do NOT imply one unbroken world: the counter runs
        # straight through a reset. Every slot of the window - context, centre and the
        # successors imagination will grade against - must carry one episode id.
        ep = self._episode_ids[all_slots]
        episode_ok = (ep == ep[:, :1]).all(axis=1)
        # Combine
        mask = seq_ok & valid_ok & episode_ok
        center_slots = (counts[mask] - 1) % self._capacity
        candidates = center_slots.tolist()
        candidate_set = set(candidates)

        if not candidates:
            return None

        # Prioritized or uniform sampling
        if prioritized and self._sum_tree is not None and self._sum_tree.total > 0:
            selected, is_weights = self._prioritized_sample(
                candidates, candidate_set, batch_size, beta)
        else:
            # Uniform sampling
            if len(candidates) <= batch_size:
                selected = candidates
            else:
                selected = random.sample(candidates, batch_size)
            is_weights = None

        # Build context windows
        context_windows = []
        for slot in selected:
            ctx_slots = []
            for j in range(ctx_before, 0, -1):
                ctx_slots.append((slot - j) % self._capacity)
            context_windows.append(ctx_slots)

        return selected, context_windows, is_weights

    def _prioritized_sample(self, candidates: list[int], candidate_set: set,
                            batch_size: int, beta: float,
                            ) -> tuple[list[int], torch.Tensor]:
        """Stratified priority-based sampling with IS weight computation."""
        actual_b = min(batch_size, len(candidates))
        total = self._sum_tree.total
        segment = total / actual_b

        selected = []
        priorities = []
        max_retries = 10

        for i in range(actual_b):
            lo = segment * i
            hi = segment * (i + 1)

            for attempt in range(max_retries):
                value = random.uniform(lo, hi)
                leaf_idx = self._sum_tree.sample(value)
                if leaf_idx in candidate_set and leaf_idx not in selected:
                    selected.append(leaf_idx)
                    priorities.append(self._sum_tree.get(leaf_idx))
                    break
            else:
                # Fallback: pick random valid candidate not already selected
                remaining = [c for c in candidates if c not in selected]
                if remaining:
                    fallback = random.choice(remaining)
                    selected.append(fallback)
                    priorities.append(self._sum_tree.get(fallback))

        # Importance sampling weights: w_i = (1 / (N * P(i)))^beta, normalized
        n = self.size
        probs = np.array(priorities, dtype=np.float64) / max(total, 1e-8)
        probs = np.maximum(probs, 1e-10)  # avoid division by zero
        weights = (1.0 / (n * probs)) ** beta
        weights /= weights.max()  # normalize so max weight = 1.0

        is_weights = torch.tensor(weights, dtype=torch.float32)
        return selected, is_weights

    def update_priorities(self, indices: list[int], priorities: list[float],
                          epsilon: float = 0.01) -> None:
        """Update priorities for sampled slots. Applies curiosity zone cap.

        Args:
            indices: Ring slot indices to update.
            priorities: Raw priority values (typically |per_sample_mse| + epsilon).
            epsilon: Minimum priority floor.
        """
        if self._sum_tree is None:
            return

        # Apply epsilon floor
        raw = [max(p, epsilon) for p in priorities]

        # Curiosity zone cap: cap at median * 10
        if len(raw) > 1:
            med = statistics.median(raw)
            cap = med * 10.0
            capped = []
            n_capped = 0
            for p in raw:
                if p > cap:
                    capped.append(cap)
                    n_capped += 1
                else:
                    capped.append(p)
            if n_capped > 0:
                logger.debug("Priority curiosity cap activated: %d/%d samples capped "
                             "(median=%.4f, cap=%.4f)", n_capped, len(raw), med, cap)
        else:
            capped = raw

        for idx, p in zip(indices, capped):
            self._sum_tree.update(idx, p)
            if p > self._max_priority:
                self._max_priority = p

    def _check_consecutive_window(self, center: int, ctx_before: int,
                                  needed_after: int) -> bool:
        """Check that all slots in [center - ctx_before, ..., center + needed_after]
        are valid, consecutive by sequence id, and inside one episode.

        No caller under Source/Python: sample_with_context does the same three tests
        vectorised. It is kept as the per-window form and must apply the SAME tests, or
        anyone who wires it up gets a weaker check than the sampler's.
        """
        start = (center - ctx_before) % self._capacity
        total = ctx_before + needed_after  # Edges to check

        prev_slot = start
        if not self._valid[prev_slot]:
            return False
        episode = self._episode_ids[prev_slot]

        for j in range(1, total + 1):
            curr_slot = (start + j) % self._capacity
            if not self._valid[curr_slot]:
                return False
            if self._sequence_ids[curr_slot] != self._sequence_ids[prev_slot] + 1:
                return False
            if self._episode_ids[curr_slot] != episode:
                return False
            prev_slot = curr_slot

        return True

    # ──────────────────────────────────────────────────────────────────────
    # Gather
    # ──────────────────────────────────────────────────────────────────────

    def gather_batch(self, indices: list[int], *, skip_vision: bool = False) -> dict:
        """Gather a batch of raw inputs by ring slot indices.

        Returns a dict with batched tensors (all on self._device):
            vision_backbone: [B, max_patches, d_internal] (or None)
            vision_n_patches: [B] int64 (or None)
            audio: {"spectral": [B, 256, 2], "spatial": [B, 2]} (or None)
            proprioception: [B, proprio_dim] (or None)
            motor: [B, n_handles, features_per_handle] (or None)
            tactile: [B, 256, 8] (or None)
            voice: [B, voice_dim] (or None)
            motor_capability: [B, cap_dim]
            gaze_target: [B, 2]
            source_masks: list of B dicts
        """
        # Try C++ fast path (eliminates Python loop + dict overhead)
        if self._native_ops is not None and self._all_modalities_enabled:
            import time as _time
            _t0 = _time.perf_counter()
            _res = self._gather_batch_native(indices, skip_vision=skip_vision)
            _dt = (_time.perf_counter() - _t0) * 1000
            if not hasattr(self, '_native_gather_logged'):
                self._native_gather_logged = True
                # Log tensor properties from first C++ gather for Dynamo guard debugging
                _props = {k: (list(v.shape), str(v.dtype), v.requires_grad,
                              v.is_contiguous(), list(v.stride()))
                          for k, v in _res.items()
                          if isinstance(v, torch.Tensor)}
                logger.info("gather_batch_native [C++] first call: %.1fms, B=%d, "
                            "tensor_props=%s", _dt, len(indices), _props)
            elif _dt > 5.0:
                logger.info("gather_batch_native [C++]: %.1fms, B=%d", _dt, len(indices))
            return _res

        idx = torch.tensor(indices, dtype=torch.long)
        result = {}

        if self._vision_enabled and not skip_vision:
            # Return fp16 directly — autocast handles upcast in head_forward.
            # Avoids allocating a [B, 32K, 768] float32 copy (~1.5 GB per call).
            result["vision_backbone"] = self._vision_buf[idx]
            result["vision_n_patches"] = self._vision_n_patches[idx.cpu()]
        else:
            result["vision_backbone"] = None
            result["vision_n_patches"] = None

        if self._audio_enabled:
            result["audio"] = {
                "spectral": self._audio_spectral_buf[idx],
                "spatial": self._audio_spatial_buf[idx],
            }
        else:
            result["audio"] = None

        # Simple modalities (proprio, motor, tactile, voice)
        for desc in _SIMPLE_MODALITIES:
            if getattr(self, desc.enabled_attr):
                result[desc.result_key] = getattr(self, desc.buf_attr)[idx]
            else:
                result[desc.result_key] = None

        result["motor_capability"] = self._motor_capability_buf[idx]
        result["gaze_target"] = self._gaze_buf[idx]
        # which rows actually carry their outgoing action: without this flag a
        # consumer cannot tell a real zero-command from "never written"
        result["has_executed"] = self.has_executed[idx.to(self.has_executed.device)]
        result["source_masks"] = [self._source_masks[i] for i in indices]

        return result

    def _gather_batch_native(self, indices: list[int], *,
                             skip_vision: bool = False) -> dict:
        """C++ fast path for gather_batch.  Tensor gathers in C++, metadata in Python."""
        idx_tensor = torch.tensor(indices, dtype=torch.long)

        result = self._native_ops.gather_batch_native(
            self._vision_buf, self._audio_spectral_buf,
            self._audio_spatial_buf, self._proprio_buf,
            self._motor_buf, self._tactile_buf,
            self._voice_buf, self._motor_capability_buf,
            self._gaze_buf, idx_tensor,
            self._modality_present, skip_vision)

        # C++ doesn't handle vision_n_patches (CPU) or source_masks (Python dicts)
        if self._vision_enabled and not skip_vision:
            result["vision_n_patches"] = self._vision_n_patches[
                idx_tensor.cpu()]
        else:
            result["vision_backbone"] = None
            result["vision_n_patches"] = None

        # the C++ gather predates the executed-action ledger; add it here so both
        # paths return the same keys (the tensors are [B,24,9] and [B] - tiny)
        if getattr(self, "_executed_action_enabled", False):
            _d = self._executed_action_buf.device
            result["executed_action"] = self._executed_action_buf[idx_tensor.to(_d)]
            result["has_executed"] = self.has_executed[idx_tensor.to(self.has_executed.device)]
        result["source_masks"] = [self._source_masks[i] for i in indices]
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def estimate_vram_mb(self) -> float:
        """Estimate total GPU VRAM usage in MB."""
        total_bytes = 0
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                total_bytes += obj.nelement() * obj.element_size()
        # Token cache buffers (stored in dict, not direct attributes)
        for buf in self._token_bufs.values():
            if buf.is_cuda:
                total_bytes += buf.nelement() * buf.element_size()
        return total_bytes / (1024 * 1024)

    def successor_index(self, slot: int, offset: int = 1) -> int:
        """Get ring slot index for a successor frame."""
        return (slot + offset) % self._capacity
