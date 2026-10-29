# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Levels 2 and 3 of the self-model over NAMED ENTITIES (UEMA spec 11.4).

What this is for. The model can tell what feels good right now. It cannot tell what is good
to HAVE AROUND: the caregiver, a toy, the mobile. The spec says a thing joins the self when
having it around goes with the body keeping well - over minutes (Level 2) and over hours
(Level 3). This module measures that, one named thing at a time.

Why it is separate from hierarchical_self.py. That module stores WHOLE MOMENTS: one slot
holds what all six senses reported at one moment, so "the caregiver" can never be one slot,
only "moments that looked like this one". Measured on run H84 on 2026-09-19: its slots were
all filled in the run's first 218 s, no moment of today belonged to any of them, and two
thirds of the movement of their presence came from a placeholder sense. An entity here is
instead a name with its own presence signal read from the world (mimo_bridge/
entity_presence.py), so it survives the encoders changing and can be pointed at.

What it does NOT do. It is dark: it feeds no lesson, no goal and no wellness term. Turning a
proven link into something the body acts on is self_value_reader.py, behind its own gate.

Five differences from the hierarchical self, each answering a measured defect:
  1. Presence is a named thing's own signal, not a likeness between token patterns.
  2. Each body's presence is correlated with THAT body's wellness, then pooled. The
     hierarchical self averaged all 8 bodies into one number first.
  3. The running averages are corrected for starting at zero, so early links do not all read
     positive (measured on H84: all 6,400 hours-level links positive, average +0.12).
  4. A window counts for an entity only if its presence MOVED during it. The whole-moment
     table gave every slot a window by the clock.
  5. The signal it learns from is a choice: the body's wellness, or an ABILITY signal
     supplied by the caller (what the project's goal asks for). Which one was used is
     recorded in the stats and in the checkpoint.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from mimo_bridge import self_evidence as EV

log = logging.getLogger("mimo_bridge.entity_self")

MOVED_FLOOR = 0.05
"""How far an entity's presence must move away from its own recent average, at some point
inside a window, for that window to count for it. Presence runs 0 to 1 and a missing reading
is 0, so this only has to clear arithmetic noise. An entity that was never there, and one
that never left, both sit at 0 movement: neither can say anything about wellness, because a
link needs both sides."""


class EntitySelfDark(nn.Module):
    """One row per named entity, two levels, no trained weights.

    Args:
        entity_names: the names, in a fixed order. The order is part of the checkpoint.
        n_bodies: how many bodies the run steps at once.
        m_l2, m_l3: the memory of the slow average at each level. 1 / (1 - m) is the window
            in moments: 10,000 and 100,000 by default, the spec's own numbers.
        n_nulls: how many control signals each level keeps.
        teacher: "wellness" (the five-formula sum) or "ability" (whatever the caller passes
            as the ability signal). Recorded in the stats.
        null_mode: how a control is made, and both flip a CENTRED quantity. "level" takes the
            value's own slow average off and flips the sign of what is left, which is what
            ships; "deviation" flips the movement the real side already measures. "level"
            used to flip the RAW value, average included, and that made the control unable to
            win: the body's felt value sits near 3.15 and moves by 0.023 inside a window, so
            a flip moved the control by 6.30 and its variance came out 18,287 times the real
            signal's, which the score then divides by. Measured on worlds with no link at all
            it counted something in 48 judgements in 100 against the 0.135 in 100 the pass
            mark is set for. With the value centred first: 0 counted in 960 judgements on
            worlds of wandering presence and drifting value, 0 in 960 on drifting value
            alone, and 2 in 960 on white noise, while a REAL link is still found in 8 of 8
            seeds at every presence rate from a quarter of a window to three windows, at a
            score of 0.976 where chance is 0.500.
            The figure of 2.34 against 7.55 in 100 that used to justify this default is a
            whole-moment-table measurement from 2026-09-10 and was never measured here.
    """

    def __init__(self, entity_names: Sequence[str], n_bodies: int = 1,
                 m_l2: float = 0.9999, m_l3: float = 0.99999, n_nulls: int = 8,
                 teacher: str = "wellness", null_mode: str = "level",
                 bias_correct: bool = True):
        super().__init__()
        self.entity_names = list(entity_names)
        self.n_entities = len(self.entity_names)
        self.n_bodies = int(n_bodies)
        self.n_nulls = int(n_nulls)
        if teacher not in ("wellness", "ability"):
            raise ValueError("teacher must be 'wellness' or 'ability', not %r" % (teacher,))
        if null_mode not in ("level", "deviation"):
            raise ValueError("null_mode must be 'level' or 'deviation', not %r" % (null_mode,))
        self.teacher = teacher
        self.null_mode = null_mode
        # False reproduces the defect this module was built to avoid (audit finding B9): the
        # running averages then start at zero and their difference carries a start-up bias.
        # Kept so a test can show the difference rather than assert the fix from words.
        self.bias_correct = bool(bias_correct)
        self.level_momentum = {"l2": float(m_l2), "l3": float(m_l3)}
        self.rank_momentum = EV.RANK_MOMENTUM
        # The fast average is a tenth of the window, the same rule the hierarchical self
        # uses, so "up" means "up against the window" and not "up since the last moment".
        self.fast_momentum = {}
        for tag, m in self.level_momentum.items():
            if not 0.0 < m < 1.0:
                raise ValueError("entity self: %s momentum %r must lie between 0 and 1" % (tag, m))
            lin = 1.0 - 10.0 * (1.0 - m)
            self.fast_momentum[tag] = lin if lin > 0.0 else float(m) ** 10
        N, B, K = self.n_entities, self.n_bodies, self.n_nulls
        self.register_buffer("frames", torch.zeros(1, dtype=torch.long))
        self.register_buffer("presence_sum", torch.zeros(N))
        for tag in ("l2", "l3"):
            self.register_buffer(f"{tag}_p_fast", torch.zeros(B, N))
            self.register_buffer(f"{tag}_p_slow", torch.zeros(B, N))
            self.register_buffer(f"{tag}_v_fast", torch.zeros(B))
            self.register_buffer(f"{tag}_v_slow", torch.zeros(B))
            self.register_buffer(f"{tag}_cov", torch.zeros(N))
            self.register_buffer(f"{tag}_pvar", torch.zeros(N))
            self.register_buffer(f"{tag}_vvar", torch.zeros(1))
            self.register_buffer(f"{tag}_nv_fast", torch.zeros(B, K))
            self.register_buffer(f"{tag}_nv_slow", torch.zeros(B, K))
            self.register_buffer(f"{tag}_ncov", torch.zeros(K, N))
            self.register_buffer(f"{tag}_nvvar", torch.zeros(K))
            self.register_buffer(f"{tag}_rank_ema", torch.full((N,), 0.5))
            self.register_buffer(f"{tag}_ticks", torch.zeros(1, dtype=torch.long))
            self.register_buffer(f"{tag}_entity_ticks", torch.zeros(N, dtype=torch.long))
            self.register_buffer(f"{tag}_moved_in_window", torch.zeros(N))
        self._frames_cpu = 0
        self._flip_cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------------ housekeeping
    def _load_from_state_dict(self, state_dict, prefix, *a, **kw):
        """A checkpoint written with a different entity list, body count or control count
        cannot be carried across: the rows would mean other things. Those buffers restart,
        loudly. A buffer the checkpoint never had starts at its own value, also loudly."""
        own = list(self.named_buffers(recurse=False))
        restarted, absent = [], []
        for name, buf in own:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = buf.detach().clone()
                absent.append(name)
            elif state_dict[key].shape != buf.shape:
                restarted.append("%s %s->%s" % (name, tuple(state_dict[key].shape),
                                                tuple(buf.shape)))
                state_dict[key] = buf.detach().clone()
        if restarted:
            log.warning("entity self: %d buffers changed shape and restart empty (the entity "
                        "list, the body count or the control count changed): %s",
                        len(restarted), ", ".join(restarted))
        if absent:
            log.warning("entity self: %d buffers absent from the checkpoint start empty: %s",
                        len(absent), ", ".join(absent))
        super()._load_from_state_dict(state_dict, prefix, *a, **kw)
        self._frames_cpu = int(self.frames.item())
        self._flip_cache = {}

    def window(self, tag: str) -> int:
        return max(1, int(round(1.0 / (1.0 - self.level_momentum[tag]))))

    def first_evidence_tick(self) -> Optional[int]:
        return EV.first_evidence_tick(self.n_entities)

    def first_evidence_steps(self, tag: str) -> Optional[int]:
        return EV.first_evidence_steps(self.window(tag), self.n_entities)

    # ------------------------------------------------------------------ the observation
    @torch.no_grad()
    def observe(self, presence: torch.Tensor, value: torch.Tensor) -> None:
        """One moment. presence is [bodies, entities] in 0..1; value is [bodies]: the summed
        wellness, or the ability signal when the teacher is "ability". Always float32, the
        caller's autocast switched off here, so a trainer, a probe and a test all compute the
        same numbers."""
        with torch.amp.autocast(presence.device.type, enabled=False):
            self._observe(presence.float(), value.float())

    @torch.no_grad()
    def _observe(self, presence: torch.Tensor, value: torch.Tensor) -> None:
        if presence.dim() != 2 or presence.shape[1] != self.n_entities:
            raise ValueError("presence must be [bodies, %d], got %s"
                             % (self.n_entities, tuple(presence.shape)))
        if value.dim() != 1 or value.shape[0] != presence.shape[0]:
            raise ValueError("one value per body is needed: presence %s, value %s"
                             % (tuple(presence.shape), tuple(value.shape)))
        dev = self.frames.device
        p = presence.to(dev).clamp(0.0, 1.0)
        v = value.to(dev)
        B = p.shape[0]
        if B != self.n_bodies:
            raise ValueError("this tracker was built for %d bodies, got %d"
                             % (self.n_bodies, B))
        fr = self._frames_cpu
        self.presence_sum.add_(p.mean(0))
        n_upd = fr + 1                       # this moment included
        for tag, m in self.level_momentum.items():
            fm = self.fast_momentum[tag]
            p_fast, p_slow = getattr(self, f"{tag}_p_fast"), getattr(self, f"{tag}_p_slow")
            v_fast, v_slow = getattr(self, f"{tag}_v_fast"), getattr(self, f"{tag}_v_slow")
            p_fast.mul_(fm).add_((1 - fm) * p)
            p_slow.mul_(m).add_((1 - m) * p)
            v_fast.mul_(fm).add_((1 - fm) * v)
            v_slow.mul_(m).add_((1 - m) * v)
            # CORRECTED FOR STARTING AT ZERO. The fast and the slow average are short of the
            # true average by their own momentum to the power of the moments lived, and those
            # two shortfalls differ, so their difference would carry a start-up bias that has
            # nothing to do with the world (audit finding B9 of the 2026-09-19 review).
            cf = EV.bias_correction(fm, n_upd) if self.bias_correct else 1.0
            cs = EV.bias_correction(m, n_upd) if self.bias_correct else 1.0
            dp = p_fast / cf - p_slow / cs                       # [B, N]
            dv = v_fast / cf - v_slow / cs                       # [B]
            getattr(self, f"{tag}_cov").mul_(m).add_((1 - m) * (dp * dv.unsqueeze(-1)).mean(0))
            getattr(self, f"{tag}_pvar").mul_(m).add_((1 - m) * (dp * dp).mean(0))
            getattr(self, f"{tag}_vvar").mul_(m).add_((1 - m) * (dv * dv).mean().reshape(1))
            # the controls: the same machinery against a value that cannot carry a true link
            window = self.window(tag)
            block = fr // window
            cached = self._flip_cache.get(tag)
            if cached is None or cached[0] != block or cached[1].device != dev:
                cached = (block, torch.tensor(EV.flip_signs(block, self.n_nulls), device=dev))
                self._flip_cache[tag] = cached
            signs = cached[1]                                     # [K]
            nv_fast, nv_slow = getattr(self, f"{tag}_nv_fast"), getattr(self, f"{tag}_nv_slow")
            if self.null_mode == "level":
                # CENTRED BEFORE THE FLIP, and this is the whole of the control's validity.
                # Flipping the sign of the raw value moves it by twice the value's MEAN, and
                # the body's felt value sits around 3.15 while its movement inside a window is
                # 0.023. So a flip moved the control by 6.30 against a real signal of 0.023 -
                # 273 times - and the control's variance came out 18,287 times the real one's.
                # Dividing by that variance crushed the control to nothing, so the real signal
                # beat it by construction: measured on worlds with NO link at all, something
                # was counted in 48 judgements in 100 against the 0.135 in 100 this test is
                # set to allow, which is 356 times the setting. White noise alone reproduced
                # it, which places the cause here and not in the world. Subtracting the
                # value's own slow average first leaves a quantity centred on zero, which is
                # what the real side already measures, so the flip changes the SHAPE of the
                # link over time and not the size of the number.
                nv = (v - v_slow / cs).unsqueeze(-1) * signs      # [B, K]
                nv_fast.mul_(fm).add_((1 - fm) * nv)
                nv_slow.mul_(m).add_((1 - m) * nv)
                ndv = nv_fast / cf - nv_slow / cs                 # [B, K]
            else:
                ndv = dv.unsqueeze(-1) * signs                    # [B, K]
            getattr(self, f"{tag}_ncov").mul_(m).add_(
                (1 - m) * torch.einsum("bk,bn->kn", ndv, dp) / max(B, 1))
            getattr(self, f"{tag}_nvvar").mul_(m).add_((1 - m) * (ndv * ndv).mean(0))
            # DID THIS WINDOW CARRY INFORMATION ABOUT THIS ENTITY? Only if its presence
            # MOVED away from its own recent average at some point - which is the very
            # quantity the link is computed from. An entity that was never there, and one
            # that never left, both sit at 0 and are passed over. The hierarchical self
            # handed a window to every slot by the clock (audit finding B3), which is how
            # 6,400 slots each held 11 windows on run H84 while no moment belonged to any.
            getattr(self, f"{tag}_moved_in_window").copy_(
                torch.maximum(getattr(self, f"{tag}_moved_in_window"), dp.abs().max(0).values))
            if (fr + 1) % window == 0 and fr + 1 >= 2 * window:
                self._tick(tag, window)
        self.frames.add_(1)
        self._frames_cpu += 1

    @torch.no_grad()
    def _tick(self, tag: str, window: int) -> None:
        """One window boundary: score every entity against its own controls."""
        pvar = getattr(self, f"{tag}_pvar")
        vvar = getattr(self, f"{tag}_vvar")
        memb = (getattr(self, f"{tag}_cov") / (pvar * vvar).sqrt().clamp_min(1e-12)).abs()
        nden = (pvar.unsqueeze(0) * getattr(self, f"{tag}_nvvar").unsqueeze(-1)).sqrt().clamp_min(1e-12)
        nmemb = (getattr(self, f"{tag}_ncov") / nden).abs()       # [K, N]
        rank = (nmemb < memb.unsqueeze(0)).float().mean(0)        # [N]
        varied = getattr(self, f"{tag}_moved_in_window") > MOVED_FLOOR
        re = getattr(self, f"{tag}_rank_ema")
        # a window that carried nothing about an entity leaves that entity's score untouched;
        # it is neither credit nor a penalty
        re.copy_(torch.where(varied, self.rank_momentum * re + (1 - self.rank_momentum) * rank, re))
        getattr(self, f"{tag}_ticks").add_(1)
        getattr(self, f"{tag}_entity_ticks").add_(varied.long())
        getattr(self, f"{tag}_moved_in_window").zero_()

    # ------------------------------------------------------------------ what it found
    @torch.no_grad()
    def links(self, tag: str) -> torch.Tensor:
        """The signed link per entity, -1 to +1: how closely its presence and the teacher
        signal rise and fall together inside the window."""
        pvar, vvar = getattr(self, f"{tag}_pvar"), getattr(self, f"{tag}_vvar")
        return getattr(self, f"{tag}_cov") / (pvar * vvar).sqrt().clamp_min(1e-12)

    @torch.no_grad()
    def counted(self, tag: str) -> torch.Tensor:
        """Which entities have PROVEN their link: their score clears the pass mark judged on
        their own windows, they have at least 3 of them, and the level as a whole has lived
        enough windows for any score to be able to clear it. All-false while unreachable."""
        need = self.first_evidence_tick()
        out = torch.zeros(self.n_entities, dtype=torch.bool, device=self.frames.device)
        if need is None:
            return out
        own = getattr(self, f"{tag}_entity_ticks")
        if int(own.max().item()) < need:
            return out
        line = torch.tensor([EV.rank_line(max(int(t), 1), self.n_entities)
                             for t in own.tolist()], device=out.device)
        return (getattr(self, f"{tag}_rank_ema") > line) & (own >= need)

    @torch.no_grad()
    def stats(self) -> Optional[dict]:
        """Small numbers for the run log and the page. Called at log intervals only."""
        fr = int(self.frames.item())
        if fr == 0:
            return {"frames": 0, "entities": self.n_entities, "teacher": self.teacher}
        out = {"frames": fr, "entities": self.n_entities, "teacher": self.teacher,
               "null_mode": self.null_mode, "bodies": self.n_bodies,
               "names": list(self.entity_names),
               "presence_mean": [round(float(v), 5) for v in (self.presence_sum / fr).tolist()]}
        need = self.first_evidence_tick()
        out["windows_needed"] = need
        for tag in ("l2", "l3"):
            ticks = int(getattr(self, f"{tag}_ticks").item())
            own = getattr(self, f"{tag}_entity_ticks")
            link = self.links(tag)
            counted = self.counted(tag)
            reachable = need is not None and int(own.max().item()) >= need
            out[f"{tag}_ticks"] = ticks
            out[f"{tag}_window_moments"] = self.window(tag)
            out[f"{tag}_entity_windows"] = own.tolist()
            out[f"{tag}_link"] = [round(float(v), 4) for v in link.tolist()]
            out[f"{tag}_rank"] = [round(float(v), 4) for v in getattr(self, f"{tag}_rank_ema").tolist()]
            out[f"{tag}_line"] = (round(EV.rank_line(max(ticks, 1), self.n_entities), 4)
                                  if reachable else None)
            out[f"{tag}_counted"] = int(counted.sum().item()) if reachable else -1
            out[f"{tag}_counted_names"] = [n for n, c in zip(self.entity_names, counted.tolist()) if c]
            out[f"{tag}_entities_old_enough"] = int((own >= need).sum().item()) if need else 0
            # HOW FAR OFF THE FIRST JUDGEMENT IS, and how far off the LAST. The first uses
            # the entity that has lived the most of its own windows, so it says when ANY
            # thing can be judged. On its own it flatters the set: it read 50,000 steps at
            # 76,333 moments lived while the squeaker, which had 2 of the 7 windows it needs,
            # was about 315,000 steps away. A thing is only ever judged on ITS OWN windows,
            # and a rarely present thing lives them slowly, so both numbers are reported.
            out[f"{tag}_steps_to_first_evidence"] = (
                0 if reachable else (need - int(own.max().item())) * self.window(tag))
            out[f"{tag}_steps_until_every_thing_can_be_judged"] = (
                0 if need is not None and int(own.min().item()) >= need
                else (need - int(own.min().item())) * self.window(tag) if need else None)
            out[f"{tag}_slowest_thing"] = (
                self.entity_names[int(own.argmin().item())] if self.n_entities else None)
        return out

    @torch.no_grad()
    def report_lines(self) -> List[str]:
        """One line per entity, for a log: name, presence, link, score, windows, counted."""
        st = self.stats() or {}
        if not st.get("frames"):
            return ["entity self: nothing observed yet"]
        lines = []
        for i, name in enumerate(self.entity_names):
            lines.append(
                "  %-18s present %.3f of the time | minutes link %+.3f (score %.3f of a pass "
                "mark %s, %d windows of its own)%s"
                % (name, st["presence_mean"][i], st["l2_link"][i], st["l2_rank"][i],
                   "not reachable yet" if st["l2_line"] is None else "%.3f" % st["l2_line"],
                   st["l2_entity_windows"][i],
                   " COUNTED" if name in st["l2_counted_names"] else ""))
        return lines
