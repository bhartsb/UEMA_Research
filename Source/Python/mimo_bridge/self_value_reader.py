# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The reader: it turns a PROVEN entity link into something the body acts to keep.

The self-model's slow levels (entity_self.py) only watch. This is the step the spec calls
for and the code never had: "It learns that certain states correlate with sustained
wellness, and the gradient maintains them" (UEMA 11.4). Here that is made concrete and
narrow.

WHAT IT DOES. The planner keeps a ring of remembered moments and steers the body toward the
one it values most. For each remembered moment this reader adds

    bonus = weight x sum over counted entities of (link of that entity) x (its presence then)

THE LINK CARRIES ITS SIGN. It is how a thing's presence moves with the body's value, from
-1 to +1. A thing whose presence goes with the body feeling BETTER has a link above zero, so a
moment it was present is worth more and the planner picks that moment as a goal; losing the
thing lowers the value of what the body can reach and the planner works to get it back. A thing
whose presence goes with the body feeling WORSE has a link below zero, so a moment it was
present is worth less and the planner aims elsewhere. Both are the same prediction and both are
acted on. The evidence test proves the link is REAL, judging the size of the correlation and
not its direction, and 4 of the 9 links measured on the live run are below zero. Nothing is
told to protect anything, and nothing is told which direction to prefer.

WHAT KEEPS IT ACCEPTABLE. Three things, all required before a single unit of value is added:
  1. Only entities whose link has PASSED the evidence test are read; nothing else counts.
WHAT THE GATE PROVES, AND WHAT IT DOES NOT. It proves the value is PREDICTIVE and that WHICH
thing carries the link matters: the real reader tracks the body's value and the same link
weights on other things do not. It does not prove the thing CAUSES the value. A world where a
third cause moves the body's value and one thing's presence follows that same cause passes,
and passes correctly, because that thing's presence really does say what the body will feel
and the other things' presence does not. Measured 2026-09-21 on exactly such a world: the real
reader tracked at 0.845 and the strongest of eight shuffles at 0.242. "The toy is around when
I feel good" is what is earned; "the toy makes me feel good" is not.

  2. EIGHT SHUFFLED CONTROLS run beside the real reader on every moment. Each holds the same
     link values attached to the wrong entities (a derangement: no entity keeps its own). The
     real reader must predict the body's wellness better than the STRONGEST of the eight,
     which is judged on the size of its correlation and not its direction: a shuffle that
     moves opposite to wellness has found just as much structure as one that moves with it.
     A shuffle whose own value never moved is not counted, because it was never a test. If
     the real reader cannot beat them, the link values carry nothing about WHICH entity
     matters, and the gate stays shut.
  3. The gate must stay open for a run of windows, not one lucky one.
The gate is reported every window, whether it is open or shut and why, so the operator never
has to infer it.

WHAT IT IS NOT. It is not a reward. Nothing is scored for reaching the goal, and the bonus
never enters a loss. It changes which remembered moment the planner aims at, and that is all.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

log = logging.getLogger("mimo_bridge.self_value")


def derangement(n: int, seed: int = 0) -> List[int]:
    """A fixed shuffle in which nothing keeps its own place. For n < 2 there is none, and the
    caller gets the identity back - with 0 or 1 entities the control cannot say anything."""
    if n < 2:
        return list(range(n))
    g = torch.Generator().manual_seed(int(seed))
    for _ in range(64):
        perm = torch.randperm(n, generator=g).tolist()
        if all(perm[i] != i for i in range(n)):
            return perm
    return [(i + 1) % n for i in range(n)]      # the plain rotation always deranges


class SelfValueReader(nn.Module):
    """Reads the entity self's proven links into a value the planner can act on.

    Args:
        entity_names: the same names, in the same order, as the tracker it reads.
        weight: how much one unit of link-times-presence is worth against the body's own
            wellness, whose five terms each run 0 to 1. 0 keeps the reader watching only.
        level: "l2" (minutes) or "l3" (hours), the level whose links are read.
        seed: picks the control's shuffle.
        gate_windows: how many windows in a row the real reader must beat the shuffled one
            before any value is added.
        margin: how far ahead of its control the real reader must be, as a correlation
            (-1 to +1). 0.05 is above the spread of a correlation on the moments one window
            holds: with 800 body-moments an unrelated pair reads about 1 / sqrt(800) = 0.035,
            and the run of windows below carries what that single number cannot.
        min_correlation: how well the real reader must track wellness on its own, whatever
            its control does. 0.1 is about three times that spread.
    """

    def __init__(self, entity_names: Sequence[str], weight: float = 0.0, level: str = "l2",
                 seed: int = 0, gate_windows: int = 3, margin: float = 0.05,
                 min_correlation: float = 0.1, n_controls: int = 8):
        super().__init__()
        if level not in ("l2", "l3"):
            raise ValueError("level must be 'l2' or 'l3', not %r" % (level,))
        self.entity_names = list(entity_names)
        self.n_entities = len(self.entity_names)
        self.weight = float(weight)
        self.level = level
        self.gate_windows = int(gate_windows)
        self.margin = float(margin)
        self.min_correlation = float(min_correlation)
        self.n_controls = int(n_controls)
        # EIGHT SHUFFLES, NOT ONE. With a single shuffle the control is one draw: which entity
        # lands on which is decided once, and whether it happens to carry structure is luck.
        # Measured 2026-09-21 on worlds where wellness was driven by a third cause that the
        # counted entity's presence merely followed, so nothing entity-specific existed to
        # find: the gate opened at the fewest windows it allows in 4 placements out of 4. One
        # of the shuffles in use put a thing present 1.4% of the time against a thing present
        # 0.42% of the time, and another paired the two critters, which the presence report
        # makes opposite to each other by construction. The real reader must now beat the BEST
        # of eight shuffles.
        self.perms = [derangement(self.n_entities, seed + 7919 * k)
                      for k in range(self.n_controls)]
        self.perm = self.perms[0]           # kept: the first shuffle, for anything that reads it
        N = self.n_entities
        self.register_buffer("link", torch.zeros(N))          # signed link per entity
        self.register_buffer("counted", torch.zeros(N))       # 1 where the link has passed
        self.register_buffer("refreshed_at", torch.zeros(1, dtype=torch.long))
        # the running comparison of the real reader against its shuffled control
        self.register_buffer("cmp_n", torch.zeros(1, dtype=torch.long))
        # running sums over the moments since the last window, enough for one correlation per
        # reader: the real one and every shuffle. Laid out as the real reader's sum, its sum
        # times wellness and its sum of squares; then the same three for each shuffle; then
        # the wellness sum and its sum of squares.
        self.register_buffer("cmp_sums", torch.zeros(3 + 3 * self.n_controls + 2))
        self.register_buffer("gate_streak", torch.zeros(1, dtype=torch.long))
        self.register_buffer("gate_open", torch.zeros(1, dtype=torch.bool))
        self.last_report: Dict[str, object] = {}

    # ------------------------------------------------------------------ housekeeping
    def _load_from_state_dict(self, state_dict, prefix, *a, **kw):
        """A RESUME NEVER INHERITS AN OPEN GATE. The links and the pass list are carried -
        they cost the tracker its windows to earn and nothing here discards them - but the
        gate, its run of windows and the running comparison restart, so the reader shows
        again, on the moments after the resume, that it beats its own shuffled control
        before one unit of value reaches the planner.

        The trainer's own resume path did this in four lines after calling this method
        (lem_model.load_checkpoint). Any other caller - a probe, a tool, a test - got an
        open gate with an empty comparison instead: measured 2026-09-21, a plain
        load_state_dict of a state whose gate was open read applied_weight 0.25 with
        0 body-moments compared. The rule now lives in the class that owns the gate.
        """
        was_open = bool(state_dict.get(prefix + "gate_open", torch.zeros(1)).reshape(-1)[0])
        # the running comparison restarts either way, so a state saved under a different
        # number of shuffles is dropped here rather than raising on its width
        old_sums = state_dict.get(prefix + "cmp_sums")
        if old_sums is not None and old_sums.numel() != self.cmp_sums.numel():
            state_dict = dict(state_dict)
            state_dict.pop(prefix + "cmp_sums")
            log.warning("self value reader: the saved comparison used %d shuffles and this "
                        "reader uses %d, so the running comparison restarts. The proven links "
                        "and the pass list are carried unchanged.",
                        (old_sums.numel() - 5) // 3, self.n_controls)
        super()._load_from_state_dict(state_dict, prefix, *a, **kw)
        self.gate_open.fill_(False)
        self.gate_streak.zero_()
        self.cmp_n.zero_()
        self.cmp_sums.zero_()
        self.last_report = {}
        if was_open:
            log.warning("self value reader: the loaded state's gate was OPEN; it restarts "
                        "SHUT and adds no value until the real reader has beaten its "
                        "shuffled control for %d windows again. The proven links and the "
                        "pass list are carried unchanged.", self.gate_windows)

    @torch.no_grad()
    def forget(self, why: str) -> None:
        """Give back everything read from a tracker whose evidence no longer exists.

        Called when the tracker itself restarts - a change in the number of bodies throws away
        every running average it had - so the links, the pass list, the gate and the running
        comparison all go back to nothing. Leaving them would go on adding value to the
        planner goal ring from evidence that no longer exists anywhere."""
        self.link.zero_()
        self.counted.zero_()
        self.refreshed_at.zero_()
        self.gate_open.fill_(False)
        self.gate_streak.zero_()
        self.cmp_n.zero_()
        self.cmp_sums.zero_()
        self.last_report = {}
        log.warning("self value reader: giving back every proven link and shutting the gate. "
                    "%s. No value reaches the planner until the links are earned again.", why)

    # ------------------------------------------------------------------ the links
    @torch.no_grad()
    def refresh(self, tracker) -> None:
        """Take the current links and the current pass list from the tracker. Called at a
        window boundary, never per moment, so the value the planner sees holds still inside
        a window and the comparison below is not fitted to the moments it judges."""
        if list(getattr(tracker, "entity_names", [])) != self.entity_names:
            raise ValueError("the reader and the tracker disagree about the entity list")
        self.link.copy_(tracker.links(self.level).to(self.link.device))
        self.counted.copy_(tracker.counted(self.level).float().to(self.link.device))
        self.refreshed_at.fill_(int(tracker.frames.item()))

    @torch.no_grad()
    def _weights(self) -> torch.Tensor:
        return self.link * self.counted

    @torch.no_grad()
    def _control_weights(self, k: int = 0) -> torch.Tensor:
        w = self._weights()
        return w[torch.tensor(self.perms[k], device=w.device)]

    # ------------------------------------------------------------------ the value
    @torch.no_grad()
    def raw_value(self, presence: torch.Tensor) -> torch.Tensor:
        """What the reader would add per body if the gate were open, before the weight."""
        return presence.to(self.link.device).float() @ self._weights()

    @torch.no_grad()
    def control_raw_value(self, presence: torch.Tensor, k: int = 0) -> torch.Tensor:
        return presence.to(self.link.device).float() @ self._control_weights(k)

    @torch.no_grad()
    def value(self, presence: torch.Tensor) -> torch.Tensor:
        """The value actually added to a remembered moment, per body. Zero while the gate is
        shut, so a run with the reader wired in but unproven behaves exactly as before."""
        v = self.raw_value(presence) * self.weight
        return v if bool(self.gate_open.item()) else torch.zeros_like(v)

    @property
    def applied_weight(self) -> float:
        return self.weight if bool(self.gate_open.item()) else 0.0

    # ------------------------------------------------------------------ the control
    @torch.no_grad()
    def compare(self, presence: torch.Tensor, wellness: torch.Tensor) -> None:
        """One moment of evidence for the gate: does the real reader's value track the body's
        wellness better than the shuffled one does? Both are scored on moments that arrived
        AFTER the links were last refreshed, so neither is judged on the moments that made
        it."""
        r = self.raw_value(presence).double()
        w = wellness.to(r.device).double()
        parts = [r.sum(), (r * w).sum(), (r * r).sum()]
        for k in range(self.n_controls):
            c = self.control_raw_value(presence, k).double()
            parts += [c.sum(), (c * w).sum(), (c * c).sum()]
        parts += [w.sum(), (w * w).sum()]
        self.cmp_sums.add_(torch.stack(parts).float())
        self.cmp_n.add_(int(r.numel()))

    @torch.no_grad()
    def window_tick(self, tracker=None) -> Dict[str, object]:
        """Called at each window boundary: judge the window just gathered, move the gate, and
        take fresh links. Returns the report."""
        rep = self.report()
        real, ctrl = rep.get("real_tracks_wellness"), rep.get("control_tracks_wellness")
        moved = int(rep.get("shuffles_that_moved") or 0)
        # THE REAL READER'S OWN CORRELATION MUST BE POSITIVE. Its value is a prediction of how
        # a moment will feel; one that moves opposite to wellness is systematically wrong, and
        # adding it to the goal ring would aim the planner away from the moments it should
        # want. The shuffles are judged on SIZE alone, because a shuffle that anti-predicts
        # has found just as much structure as one that predicts.
        ahead = (real is not None and ctrl is not None and real >= self.min_correlation
                 and real > ctrl + self.margin and moved >= self.n_controls // 2)
        if ahead:
            self.gate_streak.add_(1)
        else:
            self.gate_streak.zero_()
        was = bool(self.gate_open.item())
        now = bool(self.gate_streak.item() >= self.gate_windows) and self.weight > 0.0
        self.gate_open.fill_(now)
        if now != was:
            log.warning("self value reader: the gate is now %s. %s",
                        "OPEN - proven entity links are changing which moment the planner "
                        "aims at" if now else "SHUT - no value is being added",
                        rep.get("why"))
        self.cmp_n.zero_()
        self.cmp_sums.zero_()
        if tracker is not None:
            self.refresh(tracker)
        # the report above was built BEFORE this window moved the streak, so its wording
        # carried the old count and a gate that had just reached its third window said
        # "2 of the 3 needed" while opening
        streak = int(self.gate_streak.item())
        if ahead and isinstance(rep.get("why"), str) and "windows needed" in rep["why"]:
            rep["why"] = rep["why"].replace("for %d of the %d windows needed"
                                            % (streak - 1, self.gate_windows),
                                            "for %d of the %d windows needed"
                                            % (streak, self.gate_windows))
        rep = dict(rep, gate_open=now, gate_streak=streak)
        self.last_report = rep
        return rep

    # ------------------------------------------------------------------ what it is doing
    @torch.no_grad()
    def report(self) -> Dict[str, object]:
        n = int(self.cmp_n.item())
        counted = [nm for nm, c in zip(self.entity_names, self.counted.tolist()) if c]
        out: Dict[str, object] = {
            "weight": self.weight, "level": self.level, "gate_open": bool(self.gate_open.item()),
            "gate_streak": int(self.gate_streak.item()), "gate_windows_needed": self.gate_windows,
            "counted_entities": counted, "n_counted": len(counted),
            "moments_compared": n,
            "link": {nm: round(float(v), 4) for nm, v in zip(self.entity_names, self.link.tolist())},
            "applied_weight": self.applied_weight,
        }
        (out["real_tracks_wellness"], out["control_tracks_wellness"],
         out["shuffles_that_moved"]) = self._tracking()
        out["shuffles"] = self.n_controls
        if not counted:
            out["why"] = ("no entity has passed the evidence test yet, so the reader has "
                          "nothing to read")
        elif n < 30:
            out["why"] = ("waiting for %d body-moments to judge the shuffled control (has %d)"
                          % (30, n))
        elif out["real_tracks_wellness"] is None:
            # this used to repeat the "waiting" test above it and could never be reached, so a
            # window of 800 moments in which the reader's value never moved was reported as
            # "waiting for 30 body-moments (has 800)" - a different finding under the wrong name
            out["why"] = ("the value the reader assigned never moved over the moments judged, "
                          "so it cannot be compared with its shuffles: nothing the counted "
                          "entities do changed inside this window")
        elif out["shuffles_that_moved"] < self.n_controls // 2:
            out["why"] = ("only %d of the %d shuffles had a value that moved at all, so there "
                          "is not enough of a control to judge against: a shuffle whose value "
                          "never moves was never a test"
                          % (out["shuffles_that_moved"], self.n_controls))
        elif out["real_tracks_wellness"] < self.min_correlation:
            out["why"] = ("the real reader tracks wellness at %.3f, under the %.3f it must "
                          "reach on its own before its shuffles matter"
                          % (out["real_tracks_wellness"], self.min_correlation))
        elif out["real_tracks_wellness"] <= (out["control_tracks_wellness"] or 0.0) + self.margin:
            out["why"] = ("the strongest of %d shuffles tracks wellness nearly as well as the "
                          "real reader (%.3f against %.3f, needing %.3f between them), so "
                          "which entity a link belongs to carries nothing yet"
                          % (out["shuffles_that_moved"], out["control_tracks_wellness"],
                             out["real_tracks_wellness"], self.margin))
        else:
            out["why"] = ("the real reader tracks wellness better than the strongest of %d "
                          "shuffles (%.3f against %.3f) for %d of the %d windows needed"
                          % (out["shuffles_that_moved"], out["real_tracks_wellness"],
                             out["control_tracks_wellness"],
                             int(self.gate_streak.item()), self.gate_windows))
        return out

    @torch.no_grad()
    def _tracking(self):
        """How well the real reader's value tracked the body's wellness over the moments since
        the last window, and how well the BEST of the shuffles did, as correlations from -1 to
        +1: 0 = the value said nothing about wellness, 1 = it rose and fell with it exactly.
        None until 30 body-moments are in, the fewest that can tell a correlation of 0.5 from
        0 at this spread.

        THE SHUFFLE IS SCORED ON THE SIZE OF ITS CORRELATION, not its sign. A shuffle whose
        value moves OPPOSITE to wellness has found just as much structure as one that moves
        with it, and scoring it signed handed the real reader the whole margin for free:
        measured 2026-09-21, a shuffle reading -0.145 counted as -0.145 rather than as 0.145
        and the gate opened on worlds with no entity-specific structure at all.

        A SHUFFLE WHOSE VALUE NEVER MOVED IS NOT A CONTROL. Its correlation cannot be
        computed, and returning 0 for it said "measured, and it found nothing" for a shuffle
        that was never tested. Those are not counted, and how many were is reported."""
        n = int(self.cmp_n.item())
        if n < 30:
            return None, None, 0
        vals = [float(x) for x in self.cmp_sums.tolist()]
        sw, sww = vals[-2], vals[-1]
        var_w = max(sww / n - (sw / n) ** 2, 0.0)

        def moved(s, sxx):
            """Did this reader's value move at all over the window?

            THE LINE IS RELATIVE TO THE VALUE'S OWN SIZE, and a flat one was not enough.
            The running sums are kept in single precision, and a variance taken as the mean
            of the squares minus the square of the mean loses almost every digit when the
            value is constant: measured 2026-09-21, a value held at exactly 0.16 over 800
            moments left a variance residue of 8.1e-09, which is 8,100 times the 1e-12 a flat
            line tested. So all eight shuffles read as having moved and the strongest was
            handed a correlation of 0.0118 built from nothing but rounding. A value must now
            vary by more than a hundred-thousandth of its own level squared, which a residue
            of that kind cannot reach while a wiggle of one part in a hundred of the level
            clears it ten times over."""
            var_x = max(sxx / n - (s / n) ** 2, 0.0)
            return var_x, var_x > max(1e-12, 1e-5 * (s / n) ** 2)

        def corr(s, sxw, sxx):
            """None when this reader's value never moved, so it was never a test."""
            var_x, did = moved(s, sxx)
            den = math.sqrt(var_x * var_w)
            if not did or den <= 1e-12:
                return None
            return max(-1.0, min(1.0, (sxw / n - (s / n) * (sw / n)) / den))

        real = corr(vals[0], vals[1], vals[2])
        best, n_moved = None, 0
        for k in range(self.n_controls):
            c = corr(vals[3 + 3 * k], vals[4 + 3 * k], vals[5 + 3 * k])
            if c is None:
                continue
            n_moved += 1
            if best is None or abs(c) > best:
                best = abs(c)
        return (None if real is None else round(real, 4),
                None if best is None else round(best, 4), n_moved)


@torch.no_grad()
def natural_takeaway(presence: torch.Tensor, wellness: torch.Tensor,
                     names: Sequence[str], floor: float = 0.5) -> Dict[str, Dict[str, float]]:
    """The take-away test, run on what the world already does.

    The crib takes things away by itself: the adult leaves every 60 s of body time, the toys
    are put away and brought back. So for each entity, compare the body's wellness in the
    moments when it was there against the moments when it was not. A proven link should show
    the same sign here. This is a natural comparison, not an intervention: other things
    change when the adult leaves, so agreement supports the link and disagreement refutes it.

    presence [moments, entities] or [moments, bodies, entities]; wellness [moments] or
    [moments, bodies]. Returns, per entity, the mean wellness with and without it, the
    difference, and how many moments each side had.
    """
    p = presence.float()
    w = wellness.float()
    if p.dim() == 3:
        p = p.reshape(-1, p.shape[-1])
        w = w.reshape(-1)
    out: Dict[str, Dict[str, float]] = {}
    for i, nm in enumerate(names):
        here = p[:, i] > floor
        gone = p[:, i] <= floor
        n_here, n_gone = int(here.sum()), int(gone.sum())
        if n_here < 30 or n_gone < 30:
            out[nm] = {"moments_present": n_here, "moments_absent": n_gone,
                       "wellness_present": float("nan"), "wellness_absent": float("nan"),
                       "difference": float("nan")}
            continue
        a, b = float(w[here].mean()), float(w[gone].mean())
        out[nm] = {"moments_present": n_here, "moments_absent": n_gone,
                   "wellness_present": round(a, 5), "wellness_absent": round(b, 5),
                   "difference": round(a - b, 5)}
    return out
