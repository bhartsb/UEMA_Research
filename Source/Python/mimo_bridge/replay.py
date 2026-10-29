# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The diary (replay memory) for the bridge loop.

What it stores, per body, per step (all small, fp16, on CPU):
  * the encoded token of every sense ([d_model] each) - the moment as the model saw it
  * the ACHIEVED handle state ([n_handles, 10], real pose from physics; 26 handles under
    trunk_v1, 24 under the legacy layout, and row 23 = the eyes in both)
  * the developmental phase limits that applied
  * the OUTGOING executed order ([n_handles, 9]) - the order in force over the
    transition OUT of this moment, so it is back-filled one add later, and the
    moment keeps has_executed False until that back-fill arrives
  * beside it, the model's gaze command for that transition ([2]: azimuth and
    elevation in radians, the model's gaze_out as the world received it, before the
    world's saccade generator and gaze reflex: the controller's eye row, or the planner's
    or the self-choice's chosen row when they author the order), back-filled with the order
    (has_ctrl_gaze). The order's eye row is written from the gaze the world SENT, so the
    two split that row into the command's part and the world's. No lesson reads its
    numbers. Under --eye-row-convention physical its arrival with the order sets the eye
    row's bit of action_contract (LemLite._eye_contract_bit), and replay then studies only
    windows of such moments (the motor fixes deploy's fourth review, fix round)

What re-studying does (LemLite.replay_update): sample R random stretches of ctx+1
consecutive steps (never crossing an episode reset), run the forward model over the
stretch and grade:
  * prediction heads against the REAL next moment's tokens,
  * the motor controller against the REAL next achieved pose (true next-frame grading,
    so the motor loss grades against measured outcomes, not the model's own echo).
There was a third term, VICReg across the batch. It graded STORED tokens - fp16 ring
contents rebuilt as leaf tensors with no path to any parameter - so it trained nothing
and only inflated replay_loss; it was removed on 2026-09-06 and a test guards its
absence.

Scope notes (differences from the UE GPUReplayBuffer):
  * tokens are stored detached, so the five SENSORY encoders keep learning only from the
    live step (storing raw vision for encoder replay would cost ~1.5 MB per body per
    step; tokens cost ~4 KB). The MOTOR encoder is the exception: replay does not use the
    stored motor token at all - it re-encodes [achieved pose, outgoing executed order]
    through the live registry with gradients on, so replay trains the forward model, the
    prediction heads, the motor controller, the persistent memory's bias readout AND the
    motor encoder (measured on the H81 checkpoint's registry: the motor encoder's 285,184
    weights take gradient in one replay_update, every other encoder exactly 0);
  * memory is a ring: oldest moments are forgotten when full.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


class ReplayMemory:
    # n_handles has no body-independent right answer: the default 24 is the LEGACY layout
    # and exists for the tests that predate trunk_v1. The live ring is built from the
    # body's own count - mimo_bridge/lem_model.py enable_replay passes
    # n_handles=len(self.handle_names), which is 26 under trunk_v1.
    def __init__(self, modalities: Sequence[str], d_model: int, n_handles: int = 24,
                 size_per_env: int = 6000, n_envs: int = 1, memory_size: int = 0, self_size: int = 0):
        self.mods = list(modalities)
        self.d = d_model
        self.nh = n_handles
        self.size = int(size_per_env)
        self.n_envs = n_envs
        self.tokens = torch.zeros(n_envs, self.size, len(self.mods), d_model, dtype=torch.float16)
        self.achieved = torch.zeros(n_envs, self.size, n_handles, 10, dtype=torch.float16)
        self.rom = torch.zeros(n_envs, self.size, dtype=torch.float16)
        self.strength = torch.zeros(n_envs, self.size, dtype=torch.float16)
        self.episode = torch.zeros(n_envs, self.size, dtype=torch.int32)   # windows must not cross resets
        # 0: independent world targets; 1: a feasible wrist-relative hand goal.
        self.action_contract = torch.zeros(n_envs, self.size, dtype=torch.int8)
        # TransitionV1.valid: 1 where the modality was REAL at that moment. Placeholder
        # (synthetic) and stale tokens must not be graded as prediction targets - the
        # imagination loss had the same gap and chased Gaussian noise for two channels.
        self.valid = torch.zeros(n_envs, self.size, len(self.mods), dtype=torch.bool)
        # TransitionV1.executed_action: the OUTGOING order - the one selected from this
        # moment that carries the body to the next one. It is not known when the moment is
        # stored (the model has not chosen it yet), so it is BACK-FILLED one add later:
        # the action that arrives with moment i is the action that left moment i-1.
        # Storing the arrival action here instead would label every transition with the
        # action that had already finished acting, which is the defect this fixes.
        self.executed = torch.zeros(n_envs, self.size, n_handles, 9, dtype=torch.float16)
        self.has_executed = torch.zeros(n_envs, self.size, dtype=torch.bool)
        # the model's gaze command beside each OUTGOING order (azimuth, elevation in radians: the
        # model's gaze_out as the world received it, before the saccade generator and the gaze reflex),
        # back-filled with the order. False where none arrived: a ring restored from a companion saved
        # before this was recorded keeps these zeros and False.
        self.ctrl_gaze = torch.zeros(n_envs, self.size, 2, dtype=torch.float32)
        self.has_ctrl_gaze = torch.zeros(n_envs, self.size, dtype=torch.bool)
        self.ptr = 0
        self.count = 0
        self._episode_id = 0
        self.memory_states = (torch.zeros(n_envs,self.size,memory_size,dtype=torch.float32)
                              if memory_size else None)
        self.self_states = (torch.zeros(n_envs,self.size,self_size,dtype=torch.float32)
                            if self_size else None)

    def mark_reset(self):
        self._episode_id += 1

    def add(self, tokens: Dict[str, torch.Tensor], achieved: torch.Tensor, dev: dict,
            source_mask: Optional[Dict[str, str]] = None,
            executed: Optional[torch.Tensor] = None, memory_state: Optional[torch.Tensor] = None,
            self_state: Optional[torch.Tensor] = None, action_contract: int = 0,
            ctrl_gaze: Optional[torch.Tensor] = None):
        """tokens: {mod: [B, 1, d]} (detached); achieved: [B, n_handles, 10] (the frame's motor slice).

        source_mask: {mod: "real"|"synthetic"|"stale"} for THIS moment. Missing means real,
        which is the pre-ledger behaviour.
        executed: [B, n_handles, 9] the order actually in force over this transition.
        ctrl_gaze: [B, 2] the model's gaze command over the same transition (azimuth, elevation,
        radians, before the world's saccade generator and gaze reflex); stored beside `executed`.
        """
        i = self.ptr % self.size
        if self.self_states is not None:
            if self_state is None or self_state.shape != self.self_states[:,i].shape:
                raise ValueError('replay requires the self perception of the sampled body')
            self.self_states[:,i] = self_state.detach().to('cpu',torch.float32)
        if self.memory_states is not None:
            if memory_state is None or memory_state.shape != self.memory_states[:,i].shape:
                raise ValueError('replay requires the memory of this exact body and observation')
            self.memory_states[:,i] = memory_state.detach().to('cpu',torch.float32)
        for k, m in enumerate(self.mods):
            self.tokens[:, i, k] = tokens[m].detach().squeeze(1).to("cpu", torch.float16)
        self.achieved[:, i] = achieved.detach().to("cpu", torch.float16)
        self.rom[:, i] = float(dev.get("rom_limit", 1.0))
        self.strength[:, i] = float(dev.get("strength_limit", 1.0))
        self.episode[:, i] = self._episode_id
        for k, m in enumerate(self.mods):
            src = source_mask.get(m, "real") if source_mask else "real"
            self.valid[:, i, k] = (str(getattr(src, "value", src)) == "real")
        self.has_executed[:, i] = False          # this moment's OUTGOING action is unknown
        has_cg = getattr(self, "has_ctrl_gaze", None)
        if has_cg is not None:
            has_cg[:, i] = False
        if executed is not None and self.ptr > 0:
            prev = (self.ptr - 1) % self.size
            # only back-fill inside the same episode: an action cannot carry a body across
            # a teleport, and labelling the pre-reset moment with the post-reset action
            # would manufacture exactly the false cause this contract exists to remove
            same_ep = self.episode[:, prev] == self._episode_id
            if bool(same_ep.any()):
                e = executed.detach().to("cpu", torch.float16)
                idx = same_ep.nonzero(as_tuple=True)[0]
                self.executed[idx, prev] = e[idx]
                self.has_executed[idx, prev] = True
                self.action_contract[idx, prev] = int(action_contract)
                if ctrl_gaze is not None and has_cg is not None:
                    g = torch.as_tensor(ctrl_gaze).detach().to("cpu", torch.float32).reshape(-1, 2)
                    if g.shape[0] != self.n_envs:
                        raise ValueError("ctrl_gaze holds %d bodies, the ring %d" % (g.shape[0], self.n_envs))
                    self.ctrl_gaze[idx, prev] = g[idx]
                    has_cg[idx, prev] = True
        self.ptr += 1
        self.count = min(self.count + 1, self.size)

    def sample_windows(self, batch: int, ctx_len: int, device, action_contract=None) -> Optional[dict]:
        """batch windows of ctx_len+1 consecutive steps from random bodies, same episode."""
        need = ctx_len + 1
        if self.count < need + 2:
            return None
        starts_tok, starts_ach, roms, strengths = [], [], [], []
        starts_val, starts_exe, starts_hex = [], [], []
        starts_memory = []
        starts_self = []
        # each kept window's body and first ring slot, as chosen below (the comparison records, record c: which
        # bodies and model steps a replay pass studied); bookkeeping only, no draw of its own
        source_env, source_slot = [], []
        tries = 0
        newest = (self.ptr - 1) % self.size
        eligible = None
        if action_contract is not None:
            starts = torch.arange(self.count - need)
            if self.count == self.size:
                starts = (starts + self.ptr) % self.size
            idx_all = (starts[:, None] + torch.arange(need)[None, :]) % self.size
            contract_ok = (self.action_contract[:, idx_all[:, :-1]] == int(action_contract)).all(-1)
            labelled = self.has_executed[:, idx_all[:, :-1]].all(-1)
            episodes = self.episode[:, idx_all]
            same_episode = (episodes == episodes[..., :1]).all(-1)
            eligible = (contract_ok & labelled & same_episode).nonzero()
            if not len(eligible):
                return None
        while len(starts_tok) < batch and tries < batch * 20:
            tries += 1
            env = int(torch.randint(self.n_envs, (1,)))
            # a valid start: window fully inside written history and not spanning the write head
            s = int(torch.randint(self.count - need, (1,)))
            if eligible is not None:
                chosen = eligible[int(torch.randint(len(eligible), (1,)))]
                env, s = int(chosen[0]), int(chosen[1])
            if self.count == self.size:                    # ring is full: index relative to oldest
                # randint(count - need) draws offsets 0..size-need-1 from the OLDEST slot,
                # so the furthest a window can reach is offset size-2 and the write head
                # at offset size-1 already lies outside every window. The test below is
                # therefore unreachable today (enumerated: 3,776 starts over all 64 write
                # head positions, 0 firings, and the newest slot appears in 0 windows) and
                # is kept as the guard that catches a widened start range. The predicate
                # must stay idx[:-1], the window's INTERIOR: a window that ENDS on the
                # newest slot is still a run of consecutive moments and is legitimate.
                s = (self.ptr + s) % self.size
                idx = [(s + j) % self.size for j in range(need)]
                if newest in idx[:-1]:                     # would span the write head
                    continue
            else:
                idx = list(range(s, s + need))
            ep = self.episode[env, idx]
            if not bool((ep == ep[0]).all()):
                continue
            source_env.append(env)
            source_slot.append(idx[0])
            starts_tok.append(self.tokens[env, idx])       # [need, n_mod, d]
            starts_ach.append(self.achieved[env, idx])     # [need, n_handles, 10]
            starts_val.append(self.valid[env, idx])        # [need, n_mod]
            starts_exe.append(self.executed[env, idx])     # [need, n_handles, 9]
            starts_hex.append(self.has_executed[env, idx]) # [need]
            roms.append(self.rom[env, idx[-1]])
            strengths.append(self.strength[env, idx[-1]])
            if self.memory_states is not None:
                starts_memory.append(self.memory_states[env,idx[-2]])
            if self.self_states is not None:
                starts_self.append(self.self_states[env,idx[-2]])
        if len(starts_tok) < max(2, batch // 4):
            return None
        return {
            'memory_state': torch.stack(starts_memory).to(device) if starts_memory else None,
            'self_state': torch.stack(starts_self).to(device) if starts_self else None,
            "tokens": torch.stack(starts_tok).to(device, torch.float32),     # [R, ctx+1, n_mod, d]
            "achieved": torch.stack(starts_ach).to(device, torch.float32),   # [R, ctx+1, n_handles, 10]
            "rom": torch.stack(roms).to(device, torch.float32),
            "strength": torch.stack(strengths).to(device, torch.float32),
            "valid": torch.stack(starts_val).to(device),                     # [R, ctx+1, n_mod] bool
            "executed": torch.stack(starts_exe).to(device, torch.float32),   # [R, ctx+1, n_handles, 9]
            "has_executed": torch.stack(starts_hex).to(device),              # [R, ctx+1] bool
            "source_env": list(source_env),                                  # [R] the body of each window
            "source_slot": list(source_slot),                                # [R] its first entry's ring slot
        }

    def clear(self):
        """Drop every stored moment. Used when the meaning of what is stored changes -
        transitions labelled with an action that was never executed teach the wrong
        contract, so the ring is emptied and refilled before action-conditioned training
        resumes (plan item 5)."""
        self.ptr = 0
        self.count = 0
        self._episode_id += 1

    def labelled_count(self) -> int:
        """Stored moments that carry their OUTGOING action, i.e. usable as the step-from
        end of an action-conditioned transition."""
        n = self.count if self.count < self.size else self.size
        return int(self.has_executed[:, :n].any(dim=0).sum())

    def fill_fraction(self) -> float:
        return self.count / float(self.size)

    def clean_fraction(self) -> float:
        """Share of stored moments carrying their OUTGOING action. The newest moment never
        does - its action has not been chosen yet - so this tops out just below 1.0."""
        if self.count == 0:
            return 0.0
        n = self.count if self.count < self.size else self.size
        if self.count == self.size:
            return float(self.has_executed.float().mean())
        return float(self.has_executed[:, :n].float().mean())
