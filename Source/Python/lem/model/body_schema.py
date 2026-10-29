# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
DifferentiableBodySchema — EMA-tracked motor-sensor correlation.

UEMA Spec Section 11: The Dynamic Self-Model.

Tracks per-handle cross-handle correlation between motor command changes and
proprioceptive sensor changes via a [n_handles, n_handles * n_proprio_per_handle]
correlation matrix. Provides:
  - coherence_score(): scalar [0,1] for wellness coherence evaluator
  - self_membership(): per-handle [0,1] self-membership vector
  - incorporation_strength(handle_indices): mean incorporation for selected handles

~6.6K elements in EMA buffers (~26 KB, no trainable weights, not in optimizer).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DifferentiableBodySchema(nn.Module):
    """EMA-tracked cross-handle motor-sensor correlation per physics handle."""

    def __init__(self, n_handles: int = 24, n_proprio_per_handle: int = 11,
                 motor_dims: int = 9, temperature: float = 5.0,
                 ema_momentum: float = 0.99, object_names: tuple = (),
                 object_dims: int = 3):
        """
        Args:
            n_handles: Number of physics handles (24 by default; handle 23 = gaze).
            n_proprio_per_handle: Useful proprio features per handle (11 =
                quat4 + pos3 + vel3 + fatigue1, excluding constant bone_index).
            motor_dims: Motor output dimensions per handle (9 = pos3 + rot4 +
                radius1 + strength1).
            temperature: Sharpness of self/world boundary in sigmoid.
                Higher = crisper classification. sigmoid(T * (score - 0.5)):
                scores > 0.5 → self, scores < 0.5 → world.
            ema_momentum: EMA decay for correlation scores. 0.99 = ~1-2s
                timescale at 60 Hz.
            object_names: outside things that may cross into the self (UEMA 11.2), in a fixed
                order - a toy, the mobile. Each gets its own sensor columns, so it is scored
                the way a handle is scored: does it move when I command? Empty keeps the map
                exactly as it was, over the body's handles alone.
            object_dims: numbers per object per moment. 3 = its position in world metres.
        """
        super().__init__()
        self.n_handles = n_handles
        self.n_proprio_per_handle = n_proprio_per_handle
        # WHERE A HANDLE IS, IN WORLD METRES, is NOT read out of proprioception. A
        # proprioception row carries a handle's position as (world cm - wrist_L world cm) /
        # 200 (mimo_bridge.datasource._payload_to_frame), so it is measured from the left
        # wrist and one of its units is two metres, while an outside thing's position arrives
        # in world metres. Comparing the two directly made a perfectly carried thing score
        # 0.5 on the size term instead of 1.0, which put object_membership at exactly
        # sigmoid(0) = 0.5 and below its own "joined the self" line; and it made handle 0
        # (wrist_L) identically zero, so the left hand could never carry anything. The caller
        # passes world positions instead, and this map keeps its own previous copy.
        # A MOVEMENT SMALLER THAN THIS, IN WORLD METRES, IS NOT EVIDENCE EITHER WAY.
        # Measured on the crib scene under motor babble, 400 model steps of 0.01 s
        # (scratch probe, 2026-09-21): a hand moves a median 0.00156 m in one step and a
        # 2 kg box nobody touched moves a median 0.0 m with its largest settling twitch at
        # 0.00071 m. 0.0001 m sits 15 times below the hand's median step and above the still
        # object's median, so a moving hand always clears it and a resting thing does not.
        self.MOVED_FLOOR = 1e-4
        self.motor_dims = motor_dims
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.n_proprio_flat = n_handles * n_proprio_per_handle
        self.object_names = tuple(object_names)
        self.n_objects = len(self.object_names)
        self.object_dims = int(object_dims)
        self.n_object_flat = self.n_objects * self.object_dims
        self.n_sensor_flat = self.n_proprio_flat

        # Cross-handle correlation matrix: each handle's motor activity tracked
        # against ALL handles' proprio dimensions. Captures cross-handle effects
        # (e.g. wrist motor → elbow proprio change).
        # Start at zero: initial membership = sigmoid(T * (0 - 0.5)) ≈ 0.076.
        # Everything starts as "world" and earns its way into "self" through
        # demonstrated motor-sensor correlation.
        self.register_buffer(
            "correlation_matrix",
            torch.zeros(n_handles, self.n_proprio_flat),
        )
        # Previous frame state for computing deltas internally
        self.register_buffer("prev_motor", torch.zeros(n_handles, motor_dims))
        self.register_buffer(
            "prev_proprio",
            torch.zeros(n_handles, n_proprio_per_handle),
        )
        # Running variance for normalization (per-handle motor, per-dim proprio)
        self.register_buffer("motor_var", torch.ones(n_handles))
        self.register_buffer("proprio_var", torch.ones(self.n_proprio_flat))
        # THE OUTSIDE THINGS' BUFFERS EXIST ONLY WHEN THERE ARE ANY, so a map over the body
        # alone is the same object it always was, down to its saved state, and an older
        # checkpoint still loads into it with every key matched.
        if self.n_objects:
            self.register_buffer("prev_object", torch.zeros(self.n_objects, self.object_dims))
            # a thing that is not out, or was just put back, must not count as having moved:
            # the caller marks it and its delta is dropped for that moment
            self.register_buffer("object_seen", torch.zeros(self.n_objects))
            # HOW WELL EACH THING'S MOVEMENT MATCHES EACH HANDLE'S, 0 to 1, as a running
            # average. This is not the handle machinery: that divides every signal by its own
            # recent size, so a toy jittering by a millimetre in a random direction would score
            # as high as a toy carried by the hand. Here a moment scores 1 only when the thing
            # moved in the same direction as the handle AND by the same amount, which is what
            # being carried means.
            self.register_buffer("object_match", torch.zeros(n_handles, self.n_objects))
            # where each handle was last moment, in WORLD METRES, so a handle's own movement
            # is measured in the same unit and the same frame as a thing's
            self.register_buffer("prev_handle_pos", torch.zeros(n_handles, self.object_dims))
            self.register_buffer("handle_pos_seen", torch.zeros(1))
            # THE SAME SCORE AGAINST THE HANDLE MOVING THE OTHER WAY - the control that
            # catches the score lying. Agreement is clamped at zero below, which gives a
            # positive average even for two unrelated movements, and taking the best of many
            # handles raises it further: measured on the live run, the toy read 0.140 while
            # its own unrelated-motion null read 0.173, so the raw number was under its own
            # noise. Reversing the handle's direction leaves an unrelated movement exactly as
            # unrelated and destroys a real agreement, so what a thing scores ABOVE this is
            # what it has earned.
            self.register_buffer("object_match_null", torch.zeros(n_handles, self.n_objects))
        # Frame counter: skip delta computation on the very first frame.
        # GPU buffer kept for checkpoint save/load; CPU shadow for hot-path check
        # to avoid .item() GPU sync every frame.
        self.register_buffer("frame_count", torch.zeros(1, dtype=torch.long))
        self._frame_count_cpu: int = 0

        # Pre-allocated scratch buffers for update() — avoids torch.zeros()
        # through the CUDA caching allocator every frame.  The allocator uses
        # a per-device mutex that contends with training thread allocations,
        # causing 30ms stalls on the inference thread.
        self.register_buffer(
            "_scratch_motor_mag",
            torch.zeros(n_handles))
        self.register_buffer(
            "_scratch_proprio_delta",
            torch.zeros(n_handles, n_proprio_per_handle))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """OBJECT SCORES SAVED BEFORE THEIR CONTROL EXISTED START AGAIN FROM NOTHING.

        Until 2026-09-21 the object half compared a handle's movement taken out of
        proprioception - measured from the left wrist and divided by 200 cm - against a
        thing's movement in world metres, and it kept no control. A saved object_match from
        then was accumulated under the wrong unit and the wrong frame, and loading it beside a
        control that starts at zero would read the whole of it as earned until the control
        caught up. It is not evidence, so it is not carried: a state that has object_match
        but no object_match_null restores every other entry and leaves the object scores at
        zero, which is "nothing measured yet under the rule in force". Measured on the H84
        checkpoint at step 266,375: the carried scores summed to 4.36 over 26 handles and two
        things, against a control of exactly 0."""
        had_match = (prefix + "object_match") in state_dict
        had_control = (prefix + "object_match_null") in state_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.n_objects and had_match and not had_control:
            self.object_match.zero_()
            import logging
            logging.getLogger("lem.body_schema").warning(
                "body map: the saved object scores predate their control and were built "
                "with a handle's movement in the wrong unit and frame, so they restart at "
                "zero; every other entry of the map is carried")

    @torch.no_grad()
    def update(self, motor_commands: torch.Tensor,
               proprio_data: torch.Tensor,
               object_state: torch.Tensor = None,
               object_present: torch.Tensor = None,
               handle_world_pos: torch.Tensor = None) -> None:
        """
        Update correlation from raw motor commands, proprioception, and where any outside
        things are.

        Args:
            motor_commands: [n_handles, motor_dims] current motor output.
            proprio_data: [n_bones, >=n_proprio_per_handle] current proprio.
                Only the first n_proprio_per_handle columns are used
                (bone_index in column 11 is excluded by default).
            object_state: [n_objects, object_dims] where each outside thing is now, in world
                metres, in the order of object_names. None when the map tracks no objects.
            object_present: [n_objects] 1 where the thing is out, 0 where it is away. A thing
                that was away last moment contributes no movement this moment, so being put
                away and brought back somewhere else is not read as the body having moved it.
            handle_world_pos: [n_handles, 3] where each handle is now, in WORLD METRES - the
                same unit and the same frame as object_state, which is what makes the two
                comparable. Proprioception cannot serve here: it carries a handle's position
                measured from the left wrist and divided by 200 cm. None means this moment
                carries no evidence about outside things and they are left untouched.
        """
        n = min(motor_commands.shape[0], proprio_data.shape[0], self.n_handles)
        motor = motor_commands[:n]
        proprio = proprio_data[:n, :self.n_proprio_per_handle]

        if self._frame_count_cpu == 0:
            # First frame: store prev values, no delta available yet
            self.prev_motor[:n].copy_(motor)
            self.prev_proprio[:n].copy_(proprio)
            self.frame_count += 1
            self._frame_count_cpu += 1
            return

        # Compute frame-to-frame deltas
        motor_delta = motor - self.prev_motor[:n]       # [n, motor_dims]
        proprio_delta = proprio - self.prev_proprio[:n]  # [n, n_proprio_per_handle]

        # Store current as prev for next frame
        self.prev_motor[:n].copy_(motor)
        self.prev_proprio[:n].copy_(proprio)
        self.frame_count += 1
        self._frame_count_cpu += 1

        # Per-handle motor activity magnitude — reuse pre-allocated buffer
        motor_mag = self._scratch_motor_mag
        motor_mag.zero_()
        motor_mag[:n] = motor_delta.norm(dim=-1)

        # Flatten proprio delta across all handles for cross-handle tracking
        proprio_delta_full = self._scratch_proprio_delta
        proprio_delta_full.zero_()
        proprio_delta_full[:n] = proprio_delta
        proprio_flat = proprio_delta_full.reshape(-1)  # [n_proprio_flat]
        if self.n_objects:
            self._update_objects(handle_world_pos, object_state, object_present)

        # Update running variance via EMA
        self.motor_var.lerp_(motor_mag.square(), 1.0 - self.ema_momentum)
        self.proprio_var.lerp_(proprio_flat.square(), 1.0 - self.ema_momentum)

        # Normalize by running RMS
        eps = 1e-8
        motor_normed = motor_mag / (self.motor_var.sqrt() + eps)
        proprio_normed = proprio_flat.abs() / (self.proprio_var.sqrt() + eps)

        # Cross-handle co-activation: outer product of normalized magnitudes
        # instant_corr[h, d] = how much handle h moved × how much proprio dim d changed
        instant_corr = motor_normed.unsqueeze(1) * proprio_normed.unsqueeze(0)

        # EMA update of correlation matrix
        self.correlation_matrix.lerp_(instant_corr, 1.0 - self.ema_momentum)

    def coherence_score(self) -> torch.Tensor:
        """Classification certainty across all handles. Returns scalar [0, 1].

        High when each dimension is clearly self (membership ~1) or clearly
        world (membership ~0). Low when dimensions are ambiguous (~0.5).
        This is what the spec calls "coherence" — confidence in the self/world
        boundary, not average self-membership.
        """
        membership = self.self_membership()
        certainty = (membership - 0.5).abs() * 2.0  # 0 at 0.5, 1 at 0 or 1
        return certainty.mean()

    @torch.no_grad()
    def _update_objects(self, handle_world_pos, object_state, object_present) -> None:
        """One moment of evidence about each outside thing (UEMA 11.2).

        A thing is carried by the body when it goes WHERE the body's part went, AS FAR as it
        went. So each handle's own movement this moment is compared with each thing's: the
        cosine between the two directions (1 = the same way, 0 = at right angles, below 0
        counts as 0) times how well their distances agree (the smaller over the larger). Both
        are 0 to 1, so a moment scores 1 only for a thing travelling with that handle.

        BOTH MOVEMENTS ARE IN WORLD METRES. The handle position comes in as world metres from
        the caller rather than out of proprioception, which carries it measured from the left
        wrist and divided by 200 cm - two different frames and a factor of two apart.

        HOW FAR AGREES, NOT JUST WHICH WAY. A thing struck and sent flying goes its own
        distance and scores the ratio of the two, so batting the mobile does not read as
        carrying it. Carrying is the claim UEMA 11.2 makes, and equal displacement is what
        carrying means.

        A thing that is away, or was away last moment, scores nothing: being put back in a new
        place is not the body having moved it. A moment in which a thing did not move scores 0
        for it, so a thing has to keep earning its place - which is the slip of UEMA 11.3."""
        dev = self.object_match.device
        n_obj = self.n_objects
        obj = (torch.zeros(n_obj, self.object_dims, device=dev) if object_state is None
               else object_state[:n_obj].to(dev).float())
        here = (torch.ones(n_obj, device=dev) if object_present is None
                else object_present[:n_obj].to(dev).float())
        usable = (here * self.object_seen[:n_obj]).unsqueeze(-1)
        obj_delta = (obj - self.prev_object[:n_obj]) * usable
        self.prev_object[:n_obj].copy_(obj)
        self.object_seen[:n_obj].copy_(here)
        if handle_world_pos is None:
            # no reading of where the handles are this moment: no evidence either way, and
            # the previous reading is dropped so the next delta is not a two-moment jump
            self.handle_pos_seen.zero_()
            return
        pos = torch.zeros(self.n_handles, self.object_dims, device=dev)
        n = min(int(handle_world_pos.shape[0]), self.n_handles)
        pos[:n] = handle_world_pos[:n, :self.object_dims].to(dev).float()
        first = float(self.handle_pos_seen) < 0.5
        if first:
            # the first reading of where the handles are, or the first after a gap: there is
            # no movement to measure yet. NOT EVIDENCE, so nothing is scored and nothing
            # decays either - a moment the map could not see is not a moment the thing sat
            # still, and treating the two alike would bleed away a score that was earned.
            self.prev_handle_pos.copy_(pos)
            self.handle_pos_seen.fill_(1.0)
            return
        hand = pos - self.prev_handle_pos
        self.prev_handle_pos.copy_(pos)
        eps = 1e-9
        h_mag = hand.norm(dim=-1, keepdim=True)                       # [n_handles, 1]
        o_mag = obj_delta.norm(dim=-1, keepdim=True)                  # [n_objects, 1]
        cos = ((hand / (h_mag + eps)) @ (obj_delta / (o_mag + eps)).T).clamp(min=0.0)
        size = (torch.minimum(h_mag, o_mag.T) / (torch.maximum(h_mag, o_mag.T) + eps))
        moved = ((h_mag > self.MOVED_FLOOR) & (o_mag.T > self.MOVED_FLOOR)).float()
        self.object_match[:, :n_obj].lerp_(cos * size * moved, 1.0 - self.ema_momentum)
        # the control, scored on the same moment with the handle's direction reversed
        against = ((-hand / (h_mag + eps)) @ (obj_delta / (o_mag + eps)).T).clamp(min=0.0)
        self.object_match_null[:, :n_obj].lerp_(against * size * moved,
                                                1.0 - self.ema_momentum)

    @torch.no_grad()
    def object_membership(self, object_indices=None) -> torch.Tensor:
        """How much each outside thing belongs to the self, in the order of object_names
        (UEMA 11.2).

        A handle is scored by how much the sensors move when that handle is commanded. An
        outside thing is scored by whether it goes WHERE a handle went, AS FAR as it went (see
        _update_objects). So a toy the hand is carrying scores high; the same toy lying in the
        crib scores low, because its own small movements point nowhere in particular. Released,
        the score falls back at the map's own rate, which is the self contracting - the slip of
        UEMA 11.3.

        WHAT IS SCORED IS THE MATCH MINUS ITS OWN CONTROL, the same match against the handle
        moving the opposite way, so a number that comes only from unrelated movement reads
        nothing. Without that subtraction the live toy read 0.140 against a control of 0.173.

        THE TWO ENDS OF THIS SCALE ARE NOT 0 AND 1. It is sigmoid(temperature x (score - 0.5))
        with temperature 5, so a thing with no evidence at all reads 0.0759 and a thing carried
        perfectly reads 0.9241. Above 0.5 means the thing kept more than half the handle's
        distance in the handle's own direction, which is the line between carried and not."""
        if not self.n_objects:
            return torch.zeros(0, device=self.correlation_matrix.device)
        # the BEST-matching handle, not the average: a toy is carried by one hand, and
        # averaging over the other twenty-five handles would bury the one that holds it.
        # The control is taken from that same handle, so the pair is like for like.
        earned = (self.object_match[:, :self.n_objects]
                  - self.object_match_null[:, :self.n_objects]).clamp(min=0.0)
        scores = earned.max(dim=0).values
        if object_indices is not None:
            scores = scores[object_indices]
        return torch.sigmoid(self.temperature * (scores - 0.5))

    @torch.no_grad()
    def object_evidence(self) -> dict:
        """What each outside thing scored, what its control scored, and the difference - the
        three numbers a report needs to say whether a thing has been taken into the self or
        the reading is noise. Keyed by the thing's own name."""
        if not self.n_objects:
            return {}
        best = self.object_match[:, :self.n_objects].max(dim=0)
        out = {}
        for i, name in enumerate(self.object_names):
            h = int(best.indices[i])
            out[name] = dict(match=float(self.object_match[h, i]),
                             control=float(self.object_match_null[h, i]),
                             earned=float(max(0.0, self.object_match[h, i]
                                              - self.object_match_null[h, i])),
                             best_handle=h,
                             membership=float(self.object_membership()[i]))
        return out

    @torch.no_grad()
    def self_extent(self) -> torch.Tensor:
        """The share of everything the map watches that it calls its own, 0 to 1: the body's
        handles and any outside thing that has been taken in. This is what the coherence term
        of wellness carries, so a toy taken into the self raises wellness and losing it lowers
        wellness, with nothing told to want either."""
        parts = [self.self_membership()]
        if self.n_objects:
            parts.append(self.object_membership())
        return torch.cat(parts).mean()

    def self_membership(self, handle_indices=None) -> torch.Tensor:
        """Per-handle self-membership. Returns [n_handles] or subset in [0, 1].

        sigmoid(temperature * (score - 0.5)): scores above 0.5 map to "self",
        below 0.5 to "world". Temperature controls boundary sharpness.

        Args:
            handle_indices: Optional list/tensor of handle indices to return.
                If None, returns all handles.
        """
        # the body's own sensor columns only, so adding an outside thing cannot move a
        # handle's score
        scores = self.correlation_matrix[:, :self.n_proprio_flat].abs().mean(dim=-1)
        if handle_indices is not None:
            scores = scores[handle_indices]
        return torch.sigmoid(self.temperature * (scores - 0.5))

    def incorporation_strength(self, handle_indices: list[int]) -> torch.Tensor:
        """Mean self-membership for selected handles. Returns scalar."""
        return self.self_membership(handle_indices).mean()
