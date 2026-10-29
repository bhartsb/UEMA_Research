# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Levels 2 and 3 of the hierarchical self-model (UEMA spec 11.4), DARK mode.

Level 1 - the motor-correlation body schema - is live elsewhere. This module adds the
two slow levels: entities whose PRESENCE correlates with wellness change over minutes
(Level 2, control timescale) and hours (Level 3, abstract timescale). Nothing is named
by the operator:

  entities    64 slots formed by online EMA clustering of the model's own fused frame
              representation (every active sense's token, concatenated). Which states
              of world and body become entities is discovered, never picked. A frame
              moves only the centre of the slot it belongs to under the admission rule's
              own line (every live sense at cosine 0.9 or above); a frame that belongs
              to no slot is what that rule calls new (audit P238, 2026-09-19).
  presence    soft assignment of the current frame to the slots (cosine similarity at
              a fixed temperature; the logged assignment entropy is the control that
              would expose a degenerate temperature).
  membership  per-slot correlation between presence change and wellness change at the
              level's timescale. The spec's windowed endpoint difference
              (trajectory[-1] - trajectory[0] over ~10K / ~100K steps) is implemented
              as a BAND-PASS: fast EMA minus slow EMA, the fast window 10x shorter.
              (current - slow EMA) alone was rejected in review: it is a high-pass
              that lets per-frame jitter dominate, measuring "feels good this instant"
              rather than change at the level's window. An approximation either way,
              stated here, not hidden. Timescales are MODEL STEPS, not wall time.

DARK: this module only observes. It feeds no loss, no goal, no wellness term - turning
membership into felt value is a separate later switch that ships with its shuffled-
credit control. Everything is EMA buffers outside the gradient path: no trainable
weights, never in the optimizer, exactly like the Level 1 schema.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mimo_bridge.hself")


def _sig4(v) -> float:
    """A float cut to 4 significant digits. round(v, 7) prints 0.0 for anything under
    5e-08, which is every rotation a 6,400-slot table produces (audit P240)."""
    return float(f"{float(v):.3e}")


def _quarter_turn(one_minus_cos) -> Optional[int]:
    """Updates a centre needs to turn 90 degrees if every update turned it the same way
    by this much (the per-update rotation is given as 1 - cos). None for no rotation."""
    d = float(one_minus_cos)
    if not d > 0.0:
        return None
    theta = 2.0 * math.asin(min(1.0, math.sqrt(d / 2.0)))
    return int(round((math.pi / 2.0) / theta))


class HierarchicalSelfDark(nn.Module):
    def __init__(self, modalities: List[str], d_model: int, n_slots: int = 64,
                 m_center: float = 0.999, m_occ: float = 0.999,
                 m_l2: float = 0.9999, m_l3: float = 0.99999,
                 temperature: float = 10.0, novelty_sim: float = 0.9):
        super().__init__()
        self.modalities = list(modalities)
        self.d_model = int(d_model)
        self.d_in = len(self.modalities) * self.d_model
        self.n_slots = int(n_slots)
        self.m_center, self.m_occ = float(m_center), float(m_occ)
        self.temperature = float(temperature)
        self.novelty_sim = float(novelty_sim)
        self.level_momentum = {"l2": float(m_l2), "l3": float(m_l3)}
        self.register_buffer("centers", torch.zeros(self.n_slots, self.d_in))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("occupancy", torch.zeros(self.n_slots))
        self.register_buffer("frames", torch.zeros(1, dtype=torch.long))
        for tag in ("l2", "l3"):
            self.register_buffer(f"{tag}_p_ema", torch.zeros(self.n_slots))
            self.register_buffer(f"{tag}_p_fast", torch.zeros(self.n_slots))
            self.register_buffer(f"{tag}_w_ema", torch.zeros(1))
            self.register_buffer(f"{tag}_w_fast", torch.zeros(1))
            self.register_buffer(f"{tag}_cov", torch.zeros(self.n_slots))
            self.register_buffer(f"{tag}_pvar", torch.full((self.n_slots,), 1e-6))
            self.register_buffer(f"{tag}_wvar", torch.full((1,), 1e-6))
        # The fast window is 10x shorter than the level's window. 1 - 10*(1 - m) is
        # that window for every m above 0.9 and is kept there, so a lineage's stored
        # p_fast / nw_fast keep the meaning they were accumulated with. At m = 0.9 it
        # reaches exactly 0.0 - the "fast average" becomes the current frame, which is
        # the (current - slow EMA) high-pass this module's docstring rejects - and
        # below 0.9 it is negative, a sign-alternating recursion that is not an average
        # of anything. m ** 10 gives the same 10x shortening and stays inside (0, 1)
        # for every valid m, so it takes over in that band, loudly.
        self.fast_momentum = {}
        for t, m in self.level_momentum.items():
            if not 0.0 < m < 1.0:
                raise ValueError(f"hself: {t} momentum {m} must lie in (0, 1)")
            lin = 1.0 - 10.0 * (1.0 - m)
            self.fast_momentum[t] = lin if lin > 0.0 else float(m) ** 10
            if lin <= 0.0:
                log.warning(
                    "hself: %s momentum %.4f (window %d frames) gives 1 - 10*(1 - m) "
                    "= %.4f, which is %s; using m**10 = %.6f instead - a fast window "
                    "of %.1f frames", t, m, int(round(1.0 / (1.0 - m))), lin,
                    "the current frame, not an average" if lin == 0.0 else
                    "negative: a sign-alternating recursion, not an average",
                    self.fast_momentum[t], 1.0 / (1.0 - self.fast_momentum[t]))
        # ENTITY STABILITY GAUGES, per centre update. center_drift is the mean 1-cos
        # across the slots that moved, center_drift_max the largest single slot's:
        # measured on the run's 6,400-slot table with a batch of 8 bodies the fastest
        # slot's step is 2,056x the mean, so a mean alone cannot see the slots that
        # move. center_step is the median step rate (eff) those numbers belong to -
        # a rotation with no step size beside it cannot be read.
        self.register_buffer("center_drift", torch.zeros(1))
        self.register_buffer("center_drift_max", torch.zeros(1))
        self.register_buffer("center_step", torch.zeros(1))
        # ASSIGNMENT CONCENTRATION: the share of a frame the top slot holds, and the
        # number of slots the frame effectively spreads over (exp of the assignment
        # entropy). Measured on this run's stored table (6,400 filled slots,
        # temperature 10, its own 4 captured frames): top share 0.0636 against a
        # uniform 0.000156, spread over 2,312 slots.
        self.register_buffer("presence_top1", torch.zeros(1))
        self.register_buffer("presence_eff", torch.zeros(1))
        # updates the two EMAs above have received, for bias correction (P221): a
        # resumed lineage's checkpoint carries no presence_* key, so the EMAs start at
        # 0 while self.frames is large; dividing by 1 - 0.999**n with THIS counter is
        # exact, and the pair reads None until 100 updates (1% of the window) exist.
        self.register_buffer("presence_n", torch.zeros(1, dtype=torch.long))
        # --- 2026-09-10 slot-capacity review repairs ---
        # per-slot birth frame (age = frames - birth; 0 for pre-upgrade slots)
        self.register_buffer("birth", torch.zeros(self.n_slots))
        # admission reasons, counted separately (review: 'test new combinations
        # separately from new single-sense features')
        self.register_buffer("admit_sense", torch.zeros(1, dtype=torch.long))
        self.register_buffer("admit_combo", torch.zeros(1, dtype=torch.long))
        # ADMISSION RATE with its denominator (audit P225): how many body-frames the
        # novelty rule examined and how many it called new, admitted or refused, full
        # table or not. On H83 ckpt_step_4097248_mechanical_v4 the gate judged 968,230
        # of 968,176 examined body-frames new (ratio 1.0001), so the refusal counts
        # alone could not size a bigger table. Both start at 0 on a resume that lacks
        # them (the absent-buffer branch below), so numerator and denominator share one
        # start.
        self.register_buffer("checked", torch.zeros(1, dtype=torch.long))
        self.register_buffer("checked_novel", torch.zeros(1, dtype=torch.long))
        # SHADOW BANK: novelty is measured even when the table is full - states
        # that WOULD have been admitted land here as prototypes with sighting
        # counts, so capacity demand is a measurement, not a guess
        self.shadow_cap = 128
        self.register_buffer("shadow_centers", torch.zeros(self.shadow_cap, self.d_in))
        self.register_buffer("shadow_hits", torch.zeros(self.shadow_cap, dtype=torch.long))
        self.register_buffer("shadow_filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("shadow_overflow", torch.zeros(1, dtype=torch.long))
        # SIGN-FLIP NULL for the significance count (2026-09-10 review: the
        # analytic 3/sqrt(frames/window) floor fell with lifetime frames while
        # the EMA estimator forgets, and calibrated neither autocorrelation
        # nor slot selection; a time-SHIFTED null was tried first and failed
        # both ways - a lag can resonate with periodic structure and read the
        # real signal as null, and with few slots the per-lag estimates spread
        # too wide to referee). This null runs the SAME band-pass correlation
        # machinery against wellness whose SIGN is flipped by a fixed
        # pseudorandom schedule once per LEVEL window (the cadence is set in
        # observe(), see the `window` line there): every alignment, periodic
        # or not, is destroyed at the band the statistic reads, and the
        # estimator dynamics survive. AMPLITUDE DOES NOT, and an earlier
        # version of this comment claimed it did: the flip multiplies the RAW
        # summed wellness LEVEL (0 to 5 across the five grounded terms, slow
        # EMA 3.79 in the live checkpoint), not its deviation, so a constant
        # level becomes a square wave of that amplitude. Measured on the
        # no-association fixture the eight null band-passed variances ran
        # 0.076 to 0.631 against a real 0.078. That ratio is not itself a bias
        # - the ranked statistic is cov/sqrt(pvar*nwvar), a correlation, which
        # is unchanged by scaling either stream, and the measured |correlation|
        # sat at 0.159 for the nulls against 0.147 for the real stream.
        # Flipping the band-passed DEVIATION instead was measured WORSE on the
        # same fixture and seeds: false alarms 2.34% -> 7.55%. What is not
        # calibrated is where the rank sits with no association present: it
        # should centre on 0.5 and measured 0.6277 on that fixture (and 0.5026
        # on a single-sense fixture), which is why stats() reports the rank's
        # measured mean and spread beside the count. The schedule is a hash of
        # the frame-block index - deterministic across resume, no stored
        # sequence.
        # WHERE THE 0.63 COMES FROM, measured 2026-09-21 on the no-association fixture of
        # test_hself_significance (64 slots, window 200, 16 windows, 4 seeds), and why it
        # stays. It is not the size of the flip: taking the value's own slow average off
        # before the flip left the score at 0.6338 against 0.6292, widened its spread from
        # 0.0792 to 0.0986 and let one of 512 slots cross the line, so it was taken back out.
        # It moves with the LENGTH of the flip block, which is what a slowly drifting value
        # predicts - the real stream has fewer independent samples inside the averaging memory
        # than a copy chopped every window, so its correlation reads larger by chance:
        #     block of 1 window: score 0.6264, planted link found in 2 of 2 seeds
        #     block of 4 windows: score 0.5805, planted link found in 0 of 2
        #     block of 12 windows: score 0.2047, planted link found in 0 of 2
        # No block length calibrates the score and keeps detection, because a long block
        # leaves the controls too few flips to differ from the real stream. The COUNT is
        # what gates, and it is exact at one window: 0 of 512 no-link slots counted and
        # every planted link found, because the family-wise line sits far enough above 0.63
        # that nothing with no link crosses it. So the design stays and the score's measured
        # middle is reported beside the count rather than applied.
        # EIGHT independent flip schedules (salted hashes): one null is a
        # single chance draw - measured on the white-noise fixture, a lone
        # null read 0.027 while the same-distribution signal drew 0.224, and
        # the floor missed. Eight independent realizations give the floor an
        # actual sample of the null distribution.
        self.n_nulls = 8
        self.register_buffer("null_valid_from", torch.zeros(1, dtype=torch.long))
        for tag in ("l2", "l3"):
            self.register_buffer(f"{tag}_nw_ema", torch.zeros(self.n_nulls))
            self.register_buffer(f"{tag}_nw_fast", torch.zeros(self.n_nulls))
            self.register_buffer(f"{tag}_ncov", torch.zeros(self.n_nulls, self.n_slots))
            self.register_buffer(f"{tag}_nwvar", torch.full((self.n_nulls,), 1e-6))
            # RANK-PERSISTENCE evidence (2026-09-10 follow-up: under slowly
            # drifting independent wellness the INSTANTANEOUS correlation
            # saturates - their 256-stream AR(1) control read |corr| up to
            # 0.99 from pure coincidence and any floor that rejects it also
            # rejects every real effect. What separates knowledge from
            # coincidence is PERSISTENCE: each level-window tick, every
            # mature slot's |correlation| is ranked against its own eight
            # nulls (rank in [0,1], 0.5 expected under no association), and
            # the rank is EMA'd over ~10 ticks. Measured on the reviewer's
            # protocol: false alarms 37 -> 5 of 256 slow-noise streams while
            # a planted effect is flagged in 256 of 256.)
            self.register_buffer(f"{tag}_rank_ema",
                                 torch.full((self.n_slots,), 0.5))
            self.register_buffer(f"{tag}_ticks", torch.zeros(1, dtype=torch.long))
            # PER-SLOT tick count: how many level-window ticks this slot was mature
            # for, which is the history behind its own rank EMA. The table-wide
            # counter above keeps counting through a slot-count growth, so a slot
            # minted later would otherwise be judged against a line built from ticks
            # it never saw - at 8 of its own ticks a perfect slot reads 0.7848 and
            # clears the table-wide line 0.7739 while its own line is 0.8062.
            self.register_buffer(f"{tag}_slot_ticks",
                                 torch.zeros(self.n_slots, dtype=torch.long))
        self.rank_momentum = 0.9        # ~10-tick memory, declared
        # HOT-PATH SHADOWS (audit P237): host copies of the scalars observe() branches on
        # every frame, so an ordinary frame makes no .item() call. The buffers stay the
        # checkpointed truth; observe() and _load_from_state_dict are the only writers of
        # either and keep each pair equal.
        self._filled_cpu = 0
        self._frames_cpu = 0
        self._null_valid_from_cpu = 0
        # the 8 null signs change once per level window, not once per frame:
        # tag -> (block index, (n_nulls,) sign tensor already on the device)
        self._flip_cache = {}
        # FRAME TAP, off unless the environment asks for it (audit P238): with HSELF_TAP_DIR and
        # HSELF_TAP_STEPS set, the first HSELF_TAP_STEPS frame batches this module observes are
        # written to that folder exactly as the table sees them - the fused unit vectors x, the
        # summed wellness w and the frame counter - so a change to the centre update can be
        # replayed against the run's own moments BEFORE a run depends on it (the style guide, "measure
        # before you demand"). It reads nothing back, holds no registered buffer (the state dict
        # keeps its 48 entries) and changes no number the module computes.
        self._live_gauges, self._live_gauges_n = None, 0     # see _observe: best_slot_cos, moments_with_a_slot
        self._tap = None
        _dir, _n = os.environ.get("HSELF_TAP_DIR"), os.environ.get("HSELF_TAP_STEPS")
        if _dir and _n and int(_n) > 0:
            self._tap = {"dir": _dir, "left": int(_n), "chunk": int(os.environ.get("HSELF_TAP_CHUNK", "500")),
                         "x": None, "w": None, "fr": [], "n": 0}
            os.makedirs(_dir, exist_ok=True)
            log.warning("hself: frame tap ON - the next %d frame batches go to %s", int(_n), _dir)

    def _tap_add(self, x, w, fr):
        tp = self._tap
        if tp["x"] is None:                      # sized by the first batch it sees
            tp["x"] = torch.empty((tp["chunk"],) + tuple(x.shape), device=x.device)
            tp["w"] = torch.empty((tp["chunk"],) + tuple(w.shape), device=w.device)
        tp["x"][tp["n"]].copy_(x); tp["w"][tp["n"]].copy_(w)
        tp["fr"].append(int(fr)); tp["n"] += 1; tp["left"] -= 1
        if tp["n"] == tp["chunk"] or tp["left"] <= 0:
            self._tap_flush()
        if tp["left"] <= 0:
            log.warning("hself: frame tap finished, files are in %s", tp["dir"])
            self._tap = None

    def _tap_flush(self):
        tp = self._tap
        if not tp or not tp["n"]:
            return
        path = os.path.join(tp["dir"], "frames_%09d_%09d.pt" % (tp["fr"][0], tp["fr"][-1]))
        torch.save({"x": tp["x"][:tp["n"]].cpu(), "w": tp["w"][:tp["n"]].cpu(),
                    "frames": torch.tensor(tp["fr"], dtype=torch.long),
                    "modalities": list(self.modalities), "d_model": self.d_model}, path + ".tmp")
        os.replace(path + ".tmp", path)
        tp["fr"], tp["n"] = [], 0

    def _load_from_state_dict(self, state_dict, prefix, *a, **kw):
        # SLOT-COUNT MIGRATION. Level-3 evidence takes 1,100,000 steps to earn (see
        # first_evidence_steps below); a resume with a different n_slots must not
        # discard it. Overlapping slots are copied into the new table and ONE summary
        # warning names the whole migration - a per-buffer line meant 15 near-identical
        # lines to read. A changed sense set (different d_in) cannot be migrated and
        # the tracker restarts. Both branches WRITE the fresh values into the incoming
        # state_dict rather than dropping keys from it, so the two production load
        # paths behave the same: lem_model.py:3320 loads with strict=False and
        # acting_snapshot.py:77 with strict=True, and a dropped key is a RuntimeError
        # under the second.
        cen_key = prefix + "centers"
        own_buffers = list(self.named_buffers(recurse=False))
        if cen_key in state_dict and state_dict[cen_key].shape[-1] != self.d_in:
            log.warning("hself: saved d_in %d != current %d (sense set changed) - "
                        "tracker restarts fresh, %d buffers reset",
                        state_dict[cen_key].shape[-1], self.d_in, len(own_buffers))
            for name, own in own_buffers:
                state_dict[prefix + name] = own.detach().clone()
        else:
            lost = self._shrink_losses(state_dict, prefix)   # read BEFORE the old tensors are sliced
            migrated, restarted, shapes = [], [], None
            for name, own in own_buffers:
                key = prefix + name
                if key in state_dict and state_dict[key].shape != own.shape:
                    old = state_dict[key]
                    if old.dim() != own.dim():
                        # a buffer whose RANK changed between versions cannot
                        # be sliced across; it restarts fresh, loudly
                        restarted.append(f"{name} {tuple(old.shape)}->"
                                         f"{tuple(own.shape)}")
                        state_dict[key] = own.detach().clone()
                        continue
                    fresh = own.detach().clone()
                    sl = tuple(slice(0, min(o, f))
                               for o, f in zip(old.shape, fresh.shape))
                    fresh[sl] = old[sl].to(fresh.dtype)
                    state_dict[key] = fresh
                    migrated.append(name)
                    if name == "centers":
                        shapes = (old.shape[0], fresh.shape[0], sl[0].stop)
            if migrated:
                o, n, carried = shapes or (0, 0, 0)
                said = ""
                if lost is not None:
                    def _lvl(v):
                        return "line not reachable yet, none earned" if v is None else str(v)
                    said = ("; %d FILLED slots discarded, of which counted as evidence: "
                            "minutes level %s, hours level %s"
                            % (lost["filled_dropped"], _lvl(lost["l2"]), _lvl(lost["l3"])))
                log.warning("hself: migrated %d buffers across a slot-count change "
                            "(%d -> %d slots allocated, %d carried, %d dropped%s): %s",
                            len(migrated), o, n, carried, max(o - carried, 0), said,
                            ", ".join(migrated))
            if restarted:
                log.warning("hself: %d buffers changed rank and restart fresh: %s",
                            len(restarted), ", ".join(restarted))
        # A checkpoint written before a gauge existed carries no key for it. Fill it
        # from this module's own init so both load paths accept it (strict=True names
        # a missing key as an error), and say which gauges start cold.
        absent = [name for name, _ in own_buffers if prefix + name not in state_dict]
        for name, own in own_buffers:
            if prefix + name not in state_dict:
                state_dict[prefix + name] = own.detach().clone()
        if absent:
            log.warning("hself: %d buffers absent from the checkpoint start at their "
                        "init: %s", len(absent), ", ".join(absent))
        # a checkpoint from before the sign-flip null existed restores frames
        # but no null EMAs: mark the null warm-up as starting NOW, or a cold
        # null would read as an absurdly quiet control
        # read off `absent`, NOT off state_dict: the loop above has just written every absent
        # key into state_dict, so a membership test here was always True and this warm-up
        # mark never fired (found 2026-09-19 while verifying audit P237)
        _had_null = "l2_ncov" not in absent
        super()._load_from_state_dict(state_dict, prefix, *a, **kw)
        self._filled_cpu = min(int(self.filled.item()), self.n_slots)
        self.filled.fill_(self._filled_cpu)
        self._frames_cpu = int(self.frames.item())
        if not _had_null:
            self.null_valid_from.fill_(self._frames_cpu)
        self._null_valid_from_cpu = int(self.null_valid_from.item())
        self._flip_cache = {}
        self._live_gauges, self._live_gauges_n = None, 0
        for tag in ("l2", "l3"):
            if f"{tag}_slot_ticks" in absent and int(self.frames.item()) > 0:
                # rebuild each slot's own tick count from the state the checkpoint
                # does carry: a slot becomes mature one level window after its birth
                # frame and then takes one tick per window, capped by the ticks the
                # table has taken at all. Without this every restored slot would look
                # newborn and no slot could be counted for 9 more of its own ticks -
                # 900,000 model steps at the hours level.
                window = max(1, int(round(1.0 / (1.0 - self.level_momentum[tag]))))
                age = (self.frames.float() - self.birth).clamp_min(0.0)
                own_ticks = (age / window).floor().long() - 1
                cap = int(getattr(self, f"{tag}_ticks").item())
                getattr(self, f"{tag}_slot_ticks").copy_(
                    own_ticks.clamp(min=0, max=max(cap, 0)))
                log.warning("hself: %s per-slot tick counts rebuilt from birth frames "
                            "(table ticks %d, slot median %d)", tag, cap,
                            int(getattr(self, f"{tag}_slot_ticks")
                                [:max(self._filled_cpu, 1)].median().item()))

    @torch.no_grad()
    def _own_slots(self, x, sim, k):
        """For each body's frame, the slot it BELONGS to: among the k filled slots that the
        admission rule would call this frame's own - every live sense at cosine novelty_sim or
        above, a silent sense (zero block) not counted - the one with the highest fused cosine.
        Returns (slot index, -1 for none; whether there is one). Fixed shapes, no host read
        (audit P237)."""
        B, M, dm = x.shape[0], len(self.modalities), self.d_model
        xr = x.reshape(B, M, dm)
        live = xr.norm(dim=-1) > 1e-6                                   # (B, M)
        cb = F.normalize(self.centers[:k].reshape(k, M, dm), dim=-1)
        ps = torch.einsum("bmd,kmd->bkm", F.normalize(xr, dim=-1), cb)  # (B, k, M)
        ok = ((ps.masked_fill(~live.unsqueeze(1), 1.0).min(-1).values >= self.novelty_sim)
              & live.any(-1, keepdim=True))                             # (B, k)
        hit = ok.any(-1)
        own = sim.masked_fill(~ok, -2.0).argmax(-1)
        return torch.where(hit, own, torch.full_like(own, -1)), hit

    @torch.no_grad()
    def observe(self, tokens: Dict[str, torch.Tensor],
                grounded: Dict[str, torch.Tensor]) -> None:
        """One frame batch: fuse the senses, assign to slots, correlate with wellness.
        Always float32: the caller's autocast is switched off HERE, so the trainer, a
        probe and a test all run the same arithmetic (audit P234 - the only guard used
        to sit at lem_model's call site, where no hself test could see it)."""
        with torch.amp.autocast(self.centers.device.type, enabled=False):
            self._observe(tokens, grounded)

    @torch.no_grad()
    def _observe(self, tokens: Dict[str, torch.Tensor],
                 grounded: Dict[str, torch.Tensor]) -> None:
        ref = next(iter(tokens.values()))
        B, dev = ref.shape[0], self.centers.device
        parts = []
        for m in self.modalities:
            v = tokens.get(m)
            parts.append(v.detach().to(dev).float().reshape(B, -1)
                         if v is not None and v.shape[0] == B
                         else torch.zeros(B, self.d_model, device=dev))
        x = F.normalize(torch.cat(parts, dim=-1), dim=-1)              # (B, d_in)
        w = torch.stack([v.detach().to(dev).float().reshape(B, -1).sum(-1)
                         for v in grounded.values()]).sum(0)           # (B,)

        fr = self._frames_cpu                       # host shadow of self.frames (P237)
        if self._tap is not None:
            self._tap_add(x, w, fr)
        if self._filled_cpu == 0:                                      # first entity
            self.centers[0] = x[0]
            self.birth[0] = float(fr)
            self._filled_cpu = 1
            self.filled.fill_(1)
        # The admission cadence rides the PERSISTED frame counter. A plain call
        # counter is the one piece of per-frame state no checkpoint carries, so its
        # phase restarted at every resume and the three frames after each restart -
        # the frames where the scene, the episode and the caregiver schedule have
        # just reset - were never admission-checked.
        if fr % 4 == 0:
            # novelty-gated seeding: a slot opens only for a state unlike every
            # existing entity (naive fill-from-the-first-frames seeded all slots
            # from one regime and the second regime never got an entity - caught
            # by test_entities_form_and_separate_regimes). Judged PER SENSE: on
            # the fused vector a sound-only event leaves 5/6 blocks unchanged
            # (cos ~0.97) and could never open a slot, yet single-sense events
            # are exactly the ones that matter (the squeak is audio-only by
            # design). A silent sense (zero block) is not novelty. Checked every
            # 4th frame to keep the sync off the hot path; entities are slow.
            #
            # 2026-09-10 capacity-review repairs, all three measured defects:
            #  - SEQUENTIAL admission: each accepted member joins the center set
            #    the NEXT member is checked against (the batch-wide check let 8
            #    identical inputs open 8 identical slots; 15 near-identical
            #    prototypes were found in the live checkpoint)
            #  - COMBINATION novelty: a state whose every sense matches SOME
            #    slot but which no single slot matches as a whole (a0+b1 after
            #    a0+b0 and a1+b1) is a new entity too - counted separately
            #  - novelty is measured EVEN WHEN FULL: refused-for-capacity
            #    states land in the shadow bank as prototypes with sighting
            #    counts, so the demand for more slots is a number, not a guess
            M, dm = len(self.modalities), self.d_model
            xb = F.normalize(x.reshape(B, M, dm), dim=-1)
            live = x.reshape(B, M, dm).norm(dim=-1) > 1e-6
            self.checked.add_(B)
            for i in range(B):
                k = self._filled_cpu
                if k == 0:
                    sense_novel, combo_novel = True, False
                else:
                    cb = F.normalize(self.centers[:k].reshape(-1, M, dm), dim=-1)
                    ps = torch.einsum("md,kmd->km", xb[i], cb)        # (k, M)
                    sense_novel = bool(((ps.max(0).values < self.novelty_sim)
                                        & live[i]).any())
                    per_center_min = ps.masked_fill(
                        ~live[i].unsqueeze(0), 1.0).min(-1).values     # (k,)
                    combo_novel = (not sense_novel and
                                   bool((per_center_min < self.novelty_sim).all())
                                   and bool(live[i].any()))
                if not (sense_novel or combo_novel):
                    continue
                self.checked_novel.add_(1)
                if self._filled_cpu < self.n_slots:
                    self.centers[self._filled_cpu] = x[i]
                    self.birth[self._filled_cpu] = float(fr)
                    self._filled_cpu += 1
                    (self.admit_sense if sense_novel else self.admit_combo).add_(1)
                else:
                    self._shadow_observe(x[i], xb[i], live[i])
            self.filled.fill_(self._filled_cpu)
        k = self._filled_cpu
        sim = x @ self.centers[:k].T                                   # (B, k)
        presence = torch.zeros(B, self.n_slots, device=dev)
        presence[:, :k] = torch.softmax(self.temperature * sim, dim=-1)
        # HOW CONCENTRATED THE ASSIGNMENT IS. The softmax argmax is the nearest centre
        # by cosine (monotone in the logit), but the centre update, the occupancy EMA
        # and the presence-wellness correlation all read the WHOLE distribution, so
        # the share the top slot holds is what says whether a moment belongs to one
        # slot: measured on this run's stored table it is 0.0636 of the frame against
        # a uniform 1/6400 = 0.000156, spread over 2,312 effective slots.
        _pk = presence[:, :k].clamp_min(1e-30)
        self.presence_top1.mul_(0.999).add_(
            0.001 * _pk.max(-1).values.mean().reshape(1))
        self.presence_eff.mul_(0.999).add_(
            0.001 * (-(_pk * _pk.log()).sum(-1)).exp().mean().reshape(1))
        self.presence_n.add_(1)

        # WHOSE CENTRE A FRAME MOVES (audit P238, 2026-09-19): only the slot the frame BELONGS to -
        # the nearest slot among those the admission rule itself would call this frame's own
        # (every live sense at cosine novelty_sim or above). A frame that belongs to no slot
        # moves nothing; it is what the admission rule calls new, and it opens a slot or is
        # counted in the shadow bank. Zero new constants: the line is the admission rule's own,
        # the step rate is still 1 - m_center.
        # HISTORY, both measured. (1) Winner-only learning with NO such line collapsed the sort:
        # on ckpt_step_3361325 one slot held 96.5% of occupancy and won the assignment on 100% of
        # probe steps, because the winning centre moved toward every frame it won, however far
        # away, and so absorbed all of ordinary experience. (2) The SOFT update that replaced it
        # on 2026-09-08 moved every slot toward every frame in proportion to its softmax share.
        # Nothing in it keeps two slots apart: with this class at its defaults a kind of moment
        # seen on 3 of 100 frames at cosine 0.70 from one seen on 97 of 100 had its slot pulled
        # onto the common one, a fresh slot opened for it and was pulled in too - 9 of 11 slots
        # on one point after 40,000 frames, and the rare kind's slot never kept a window of its
        # own. On run H84's 6,400-slot table every slot held about 1/6,400 of every moment, so
        # the same rule pulled the WHOLE table toward the middle of the data: the median cosine
        # between a slot and its nearest other slot rose from 0.9573 with the table just full
        # (step 17,424) to 0.9580 (step 120,702), 9 checkpoints, never falling by more than
        # 0.0001 - the road to the copied slots the older lineage showed at 1,000,000 frames.
        # Under the present rule a slot can absorb only frames inside the admission line, which
        # by that rule ARE its own kind of moment, so neither failure has a path (tests:
        # test_a_rare_kind_of_moment_keeps_its_own_slot, test_a_far_frame_moves_no_slot).
        # WHAT IT DOES NOT CURE: a table whose slots sit far from today's moments stays where it
        # is. On H84 the slots were opened in the first 4,268 frames, before the encoders had
        # trained; measured on 14 of the run's own moments the nearest slot sits at cosine 0.19
        # to 0.34 (1 = the same pattern), so no moment belongs to any slot and no centre moves.
        # The two gauges below (best_slot_cos, moments_with_a_slot) report exactly that, live.
        own, hit = self._own_slots(x, sim, k)                            # (B,) slot index, (B,) bool
        same = ((own.unsqueeze(1) == own.unsqueeze(0))
                & hit.unsqueeze(1) & hit.unsqueeze(0))                  # bodies that share a slot
        cnt = same.sum(1).clamp_min(1).float()                          # (B,)
        _tgt = (same.float() @ x) / cnt.unsqueeze(-1)                   # their mean frame
        safe = own.clamp_min(0)
        _cen = self.centers[safe]                                       # (B, d_in), fixed shape
        eff = torch.full((B, 1), 1.0 - self.m_center, device=dev)
        moved = F.normalize((1.0 - eff) * _cen + eff * _tgt, dim=-1)
        # ENTITY STABILITY GAUGE: membership means nothing if the entity itself is a
        # moving target, so the drift is logged and the data arbitrates m_center.
        # 1 - cos is NOT computed as 1 minus a dot product of two near-identical unit
        # vectors: float32 resolves 1 - cos only down to 2^-24 = 6.0e-08, and measured on a
        # 6,400-slot table the subtraction returned exactly 0.0 on 5,372 slots and a
        # NEGATIVE value on 645 (a rotation has no sign).
        # With c the old unit centre, u the update target and e the step rate, the
        # normalised move has cos = s / sqrt(s*s + q) for s = (1 - e) + e*(c.u) and
        # q = e^2 * (|u|^2 - (c.u)^2), so 1 - cos = q / (sqrt(s*s+q) * (sqrt(s*s+q) + s))
        # - every term O(1), no cancellation, never negative. Measured against the
        # same quantity computed entirely in float64: median relative error 1.2e-09.
        _e = eff.squeeze(-1).double()
        _cu = (_cen * _tgt).sum(-1, dtype=torch.float64)
        _q = (_e * _e) * (_tgt.pow(2).sum(-1, dtype=torch.float64)
                          - _cu * _cu).clamp_min(0.0)
        _s = (1.0 - _e) + _e * _cu
        _root = (_s * _s + _q).sqrt()
        _drift = _q / (_root * (_root + _s)).clamp_min(1e-300)
        # The three gauges advance only on a frame that moved a slot, as before ("per centre
        # update"); written without a branch so an ordinary frame asks the device for nothing
        # (audit P237). Bodies that share a slot carry that slot once (weight 1/cnt).
        _wt = hit.double() / cnt.double()
        _any = hit.any().float()
        _mean = (_drift * _wt).sum() / _wt.sum().clamp_min(1e-12)
        _max = torch.where(hit, _drift, torch.zeros_like(_drift)).max()
        self.center_drift.mul_(1.0 - 0.001 * _any).add_(0.001 * _any * _mean.float().reshape(1))
        self.center_drift_max.mul_(1.0 - 0.001 * _any).add_(0.001 * _any * _max.float().reshape(1))
        self.center_step.mul_(1.0 - 0.001 * _any).add_(0.001 * _any * (1.0 - self.m_center))
        self.centers.index_add_(0, safe, torch.where(hit.unsqueeze(-1), moved - _cen,
                                                     torch.zeros_like(_cen)) / cnt.unsqueeze(-1))
        # HOW FAR TODAY'S MOMENTS SIT FROM THE TABLE, and how many of them belong to a slot. Not
        # registered buffers (the state dict keeps its 48 entries): two running averages that
        # start again at every resume and read None until 100 frames.
        if self._live_gauges is None or self._live_gauges.device != sim.device:
            self._live_gauges, self._live_gauges_n = torch.zeros(2, device=sim.device), 0
        self._live_gauges.mul_(0.999).add_(0.001 * torch.stack(
            [sim.max(-1).values.mean(), hit.float().mean()]))
        self._live_gauges_n += 1
        p = presence.mean(0)
        self.occupancy.mul_(self.m_occ).add_((1 - self.m_occ) * p)
        wm = w.mean().reshape(1)

        # sign-flip control stream: a deterministic pseudorandom sign per
        # LEVEL-window block of frames (splitmix-style integer hash, cadence set
        # at the `window` line below), applied to this frame's wellness for the
        # null estimator only

        def _flips(block: int) -> torch.Tensor:
            out = []
            for salt in range(self.n_nulls):
                z = ((block + salt * 0x632BE59B) * 2654435761
                     + 0x9E3779B9) & 0xFFFFFFFF
                z ^= z >> 16
                z = (z * 0x45D9F3B) & 0xFFFFFFFF
                z ^= z >> 16
                out.append(1.0 if (z & 1) else -1.0)
            return torch.tensor(out)

        for tag, mom in self.level_momentum.items():
            fmom = self.fast_momentum[tag]
            p_ema = getattr(self, f"{tag}_p_ema")
            p_fast = getattr(self, f"{tag}_p_fast")
            w_ema = getattr(self, f"{tag}_w_ema")
            w_fast = getattr(self, f"{tag}_w_fast")
            p_fast.mul_(fmom).add_((1 - fmom) * p)
            w_fast.mul_(fmom).add_((1 - fmom) * wm)
            p_ema.mul_(mom).add_((1 - mom) * p)
            w_ema.mul_(mom).add_((1 - mom) * wm)
            # band-passed change at the level's window: fast average minus slow
            # average - not (current - slow), which is a high-pass that lets
            # per-frame jitter dominate the correlation
            dp, dw = p_fast - p_ema, w_fast - w_ema
            getattr(self, f"{tag}_cov").mul_(mom).add_((1 - mom) * dp * dw)
            getattr(self, f"{tag}_pvar").mul_(mom).add_((1 - mom) * dp * dp)
            getattr(self, f"{tag}_wvar").mul_(mom).add_((1 - mom) * dw * dw)
            # flip cadence = the LEVEL's window (2026-09-10 follow-up: fast-
            # window flips chopped the slow power out of the null stream, so
            # it understated the spurious level of drifting wellness - the
            # measured source of 37/256 false alarms)
            window = max(1, int(round(1.0 / (1.0 - mom))))
            block = fr // window
            cached = self._flip_cache.get(tag)
            if cached is None or cached[0] != block or cached[1].device != wm.device:
                cached = (block, _flips(block).to(wm.device))
                self._flip_cache[tag] = cached
            wl = wm * cached[1]                                 # (n_nulls,)
            nw_ema = getattr(self, f"{tag}_nw_ema")
            nw_fast = getattr(self, f"{tag}_nw_fast")
            nw_fast.mul_(fmom).add_((1 - fmom) * wl)
            nw_ema.mul_(mom).add_((1 - mom) * wl)
            ndw = (nw_fast - nw_ema).unsqueeze(-1)              # (n_nulls, 1)
            getattr(self, f"{tag}_ncov").mul_(mom).add_(
                (1 - mom) * dp.unsqueeze(0) * ndw)
            getattr(self, f"{tag}_nwvar").mul_(mom).add_(
                (1 - mom) * (ndw.squeeze(-1) ** 2))
            # rank-persistence tick, once per level window after the nulls
            # have two windows of history; YOUNG slots (age < window) sit
            # out - their cold estimates rank arbitrarily, and a new slot
            # must never lend significance to old ones (per-slot ranks also
            # remove the shared-floor coupling the review demonstrated)
            valid_from = self._null_valid_from_cpu
            if (fr + 1) % window == 0 and (fr - valid_from) >= 2 * window:
                kf = self._filled_cpu
                if kf > 0:
                    memb_abs = (getattr(self, f"{tag}_cov")[:kf]
                                / (getattr(self, f"{tag}_pvar")[:kf]
                                   * getattr(self, f"{tag}_wvar"))
                                .sqrt().clamp_min(1e-9)).abs()
                    ncov = getattr(self, f"{tag}_ncov")[:, :kf]
                    nden = (getattr(self, f"{tag}_pvar")[:kf].unsqueeze(0)
                            * getattr(self, f"{tag}_nwvar").unsqueeze(-1)
                            ).sqrt().clamp_min(1e-9)
                    nmemb_abs = (ncov / nden).abs()              # (n_nulls, kf)
                    rank = (nmemb_abs < memb_abs.unsqueeze(0)).float().mean(0)
                    mature = (fr - self.birth[:kf]) >= window
                    re = getattr(self, f"{tag}_rank_ema")
                    upd = torch.where(mature, rank, torch.full_like(rank, 0.5))
                    re[:kf] = (self.rank_momentum * re[:kf]
                               + (1 - self.rank_momentum) * upd)
                    getattr(self, f"{tag}_ticks").add_(1)
                    # P224: the per-slot tick buffer registered above is now counted,
                    # so a slot minted later is judged on its own history
                    getattr(self, f"{tag}_slot_ticks")[:kf] += mature.long()
        self.frames.add_(1)
        self._frames_cpu += 1

    @torch.no_grad()
    def _shadow_observe(self, xf: torch.Tensor, xbi: torch.Tensor,
                        livei: torch.Tensor) -> None:
        """A state that passed novelty while the table was FULL: count demand.
        Prototypes dedupe with the SAME per-sense + combination novelty rule
        the admission gate uses (2026-09-10 follow-up: the fused-cosine
        shortcut merged two patterns the admission rule holds distinct), so
        'distinct patterns turned away' and 'sightings of them' are separate,
        admission-consistent numbers."""
        s = int(self.shadow_filled.item())
        M, dm = len(self.modalities), self.d_model
        if s:
            sb = F.normalize(self.shadow_centers[:s].reshape(-1, M, dm), dim=-1)
            ps = torch.einsum("md,kmd->km", xbi, sb)
            sense_novel = bool(((ps.max(0).values < self.novelty_sim)
                                & livei).any())
            per_center_min = ps.masked_fill(
                ~livei.unsqueeze(0), 1.0).min(-1).values
            combo_novel = (not sense_novel and
                           bool((per_center_min < self.novelty_sim).all())
                           and bool(livei.any()))
            if not (sense_novel or combo_novel):
                # a repeat of a known turned-away pattern: credit its nearest
                # prototype by the admission rule's own distance
                self.shadow_hits[int(per_center_min.argmax())] += 1
                return
        if s < self.shadow_cap:
            self.shadow_centers[s] = xf
            self.shadow_hits[s] = 1
            self.shadow_filled.fill_(s + 1)
        else:
            self.shadow_overflow.add_(1)

    @torch.no_grad()
    def membership(self, tag: str) -> torch.Tensor:
        """Per-slot correlation in [-1, 1]: presence change vs wellness change."""
        cov = getattr(self, f"{tag}_cov")
        den = (getattr(self, f"{tag}_pvar") * getattr(self, f"{tag}_wvar")).sqrt()
        return cov / den.clamp_min(1e-9)

    # ---------------------------------------------------------------- pass line (P222, P224)
    RANK_ALPHA = 0.00135     # the design's own one-sided 3-sigma tail, now family-wise

    def _evidence_flags(self, rank_ema, slot_ticks, k, ticks):
        """Which of the k filled slots the pass line counts: rank EMA above the family-wise
        line judged on the slot's OWN ticks, at least 3 of them. None while the line is
        unreachable for a k-slot table. The same rule stats() counts with - a test holds
        the two equal (audit P235)."""
        k = int(k)
        if k <= 0 or int(ticks) < int(self.first_evidence_tick(k)):
            return None
        st = slot_ticks[:k].clamp_min(1)
        neff = st.float().clamp(max=1.0 / (1.0 - self.rank_momentum))
        sigma = math.sqrt(1.0 / 12.0) / neff.sqrt()
        z_k = float(torch.special.ndtri(torch.tensor(1.0 - self.RANK_ALPHA / k,
                                                     dtype=torch.float64)))
        return (rank_ema[:k] > 0.5 + z_k * sigma) & (st >= 3)

    def _shrink_losses(self, state_dict, prefix):
        """What a SMALLER table discards, read from the incoming (old) state before it is
        sliced: filled slots above the new count and, per level, how many of those the pass
        line counted as evidence (None = the line was not reachable, nothing was earned).
        None when the table is not shrinking."""
        cen, fil = state_dict.get(prefix + "centers"), state_dict.get(prefix + "filled")
        if cen is None or fil is None or cen.dim() != 2 or cen.shape[0] <= self.n_slots:
            return None
        k_old = min(int(fil.item()), cen.shape[0])
        out = {"filled_dropped": max(k_old - self.n_slots, 0), "l2": None, "l3": None}
        for tag in ("l2", "l3"):
            re, tk = state_dict.get(prefix + f"{tag}_rank_ema"), state_dict.get(prefix + f"{tag}_ticks")
            if re is None or tk is None or k_old <= self.n_slots:
                continue
            ticks = int(tk.item())
            st = state_dict.get(prefix + f"{tag}_slot_ticks")
            if st is None:       # written before per-slot ticks existed: the table's count
                st = torch.full((cen.shape[0],), ticks, dtype=torch.long, device=re.device)
            flags = self._evidence_flags(re, st, k_old, ticks)
            out[tag] = None if flags is None else int(flags[self.n_slots:].sum().item())
        return out

    def _rank_sigma(self, ticks):
        neff = min(float(ticks), 1.0 / (1.0 - self.rank_momentum))
        return math.sqrt(1.0 / 12.0) / math.sqrt(max(neff, 1e-9))

    def _rank_line(self, ticks, k=None):
        """The standing a slot must beat after `ticks` windows. k None = the legacy
        per-slot 3-sigma line (0.5 + 3*sigma). With k = the number of slots judged
        together the tail is family-wise: 0.5 + ndtri(1 - alpha/k) * sigma (P224)."""
        if k is None:
            return 0.5 + 3.0 * self._rank_sigma(ticks)
        z = float(torch.special.ndtri(torch.tensor(1.0 - self.RANK_ALPHA / max(int(k), 1),
                                                   dtype=torch.float64)))
        return 0.5 + z * self._rank_sigma(ticks)

    def _rank_ceiling(self, ticks):
        """The largest rank EMA reachable from the 0.5 seed with a rank of 1.0 at every
        tick: 0.5 + 0.5 * (1 - momentum^ticks)."""
        return 0.5 + 0.5 * (1.0 - self.rank_momentum ** int(ticks))

    def first_evidence_tick(self, k=None):
        """The reachability arithmetic for the module's own pass line: the smallest tick
        count at which a perfect slot could clear the line. 9 for the legacy line at
        momentum 0.9; 14 on a 64-slot table and 25 on a 6,400-slot table under the
        family-wise line (P222, P224). Before it, a 0 is forced by arithmetic and the
        count is reported as unmeasured (-1)."""
        for n in range(1, 100000):
            if self._rank_ceiling(n) > self._rank_line(n, k):
                return n
        return None

    def first_evidence_steps(self, tag, k=None):
        """Model steps before any slot could be counted at this level: 2 warm-up windows
        plus first_evidence_tick windows (110,000 / 1,100,000 for the legacy line)."""
        window = max(1, int(round(1.0 / (1.0 - self.level_momentum[tag]))))
        return (2 + int(self.first_evidence_tick(k))) * window

    TWIN_COS = 0.9999   # a census line, nothing is gated on it: the cosine above which two
    #                     centres are read as ONE situation stored twice (0.81 degrees apart).
    #                     It is the line the 2026-09-16 audit counted with on this lineage
    #                     (24 of 6,400 slots at frame 964,324, 25 at 1,032,873), kept so the
    #                     series stays comparable.

    @torch.no_grad()
    def _twin_census(self, k, occ, chunk=128):
        """How many DISTINCT situations the k filled slots hold (audit P238). The soft
        centre update has no term that keeps two centres apart: measured with this class
        at its defaults, a situation seen on 3 of 100 frames at cosine 0.70 from one seen
        on 97 of 100 had its slot pulled onto the common one's centre, a fresh slot opened
        for it, and that was pulled in too - 9 of 11 filled slots ended on one point after
        40,000 frames while slots_filled read 11. occ is the occupancy share over the
        filled slots. Runs at log intervals only and never copies the table: the centres
        are read in place and compared chunk x k at a time (about 14 MB of scratch at
        128 x 6,400)."""
        c = self.centers[:k]                                             # a view, no copy
        nrm = c.norm(dim=-1).clamp_min(1e-12)
        ar = torch.arange(k, device=c.device)
        nn_cos = torch.empty(k, device=c.device)
        lab = ar.clone()
        for s in range(0, k, chunk):
            sim = (c[s:s + chunk] @ c.T) / (nrm[s:s + chunk].unsqueeze(-1) * nrm.unsqueeze(0))
            lab[s:s + chunk] = torch.where(sim > self.TWIN_COS, ar, k).min(-1).values.clamp(max=k - 1)
            rows = torch.arange(sim.shape[0], device=c.device)
            sim[rows, rows + s] = -2.0                                   # not its own neighbour
            nn_cos[s:s + chunk] = sim.max(-1).values
        lab = torch.minimum(lab, ar)
        for _ in range(8):
            lab = lab[lab]                   # chains of twins settle on their lowest index
        twin = nn_cos > self.TWIN_COS
        n_twin = int(twin.sum().item())
        groups = int(lab[twin].unique().numel()) if n_twin else 0
        return {"slots_twinned": n_twin, "twin_groups": groups,
                "slots_distinct": int(k) - n_twin + groups,
                "twin_occupancy": round(float(occ[twin].sum().item()), 4),
                "nearest_other_cos_median": (round(float(nn_cos.median().item()), 4)
                                             if k > 1 else None)}

    @torch.no_grad()
    def stats(self) -> Optional[dict]:
        """Small numbers for the run log. Called at log intervals only (this syncs)."""
        if self._filled_cpu == 0:
            return {"frames": 0, "slots_filled": 0}
        # CAPACITY-INVARIANT gauges (2026-09-10 review: growing 64 -> 6,400
        # with identical evidence moved slots_live 22 -> 64 and shifted the
        # entropy, purely because allocated-but-empty slots entered the
        # denominators; every normalization below uses the FILLED count)
        k = max(self._filled_cpu, 1)
        occ = self.occupancy[:k] / self.occupancy[:k].sum().clamp_min(1e-9)
        ent = float(-(occ.clamp_min(1e-12) * occ.clamp_min(1e-12).log()).sum()
                    / math.log(max(k, 2)))
        fr = int(self.frames.item())
        births = self.birth[:k]
        out = {"frames": fr,
               "slots_filled": int(self._filled_cpu),
               "slots_live": int((occ > 0.1 / k).sum().item()),
               # how far the QUIETEST slot clears the slots_live line (P221). At 1.0151
               # on the live 6,400-slot table every slot passes by 1.5% and the count is
               # at its ceiling, which the count alone cannot show.
               "slots_live_margin": round(float(occ.min().item()) / (0.1 / k), 4),
               "occ_entropy": round(ent, 4),
               # FOUR SIGNIFICANT DIGITS, not seven decimals (audit P240): at m_center 0.999
               # one update can turn a centre by at most 1 - cos = 5.0e-07, and on a
               # 6,400-slot table the measured mean was 4.6e-12 and the fastest slot's
               # 2.7e-08 - round(x, 7) printed 0.0 for both.
               "center_drift": _sig4(self.center_drift.item()),
               "center_drift_max": _sig4(self.center_drift_max.item()),
               "center_step": _sig4(self.center_step.item()),
               # the same two rotations in lived terms: moments a centre needs for a quarter
               # turn if it keeps turning the same way at this rate. Read them against the
               # evidence windows (10,000 and 100,000 moments): a slot that turns a quarter
               # turn inside one window is a moving target for that window's membership.
               "center_quarter_turn_moments_fastest": _quarter_turn(self.center_drift_max.item()),
               "center_quarter_turn_moments_mean": _quarter_turn(self.center_drift.item()),
               # ASSIGNMENT CONCENTRATION, live (P221). Both are 0.999-rate EMA buffers
               # updated every frame and were reported nowhere. Bias-corrected by their
               # own update count (a resumed checkpoint without the keys starts both at
               # 0 while self.frames is large, so frames would be the wrong n); None
               # until 100 updates exist. (The row's "None until it clears 1.0" rule
               # never clears for a sharp assignment whose true value sits near 1.0 -
               # the EMA approaches it from below - measured on the 16-slot fixture.)
               "presence_top1": (None if int(self.presence_n.item()) < 100
                                 else round(float(self.presence_top1.item()) / (1.0 - 0.999 ** int(self.presence_n.item())), 6)),
               "presence_eff": (None if int(self.presence_n.item()) < 100
                                else round(float(self.presence_eff.item()) / (1.0 - 0.999 ** int(self.presence_n.item())), 1)),
               "slot_age_median_frames": int(torch.median(fr - births).item()),
               "slot_age_newest_frames": int((fr - births.max()).item()),
               "slot_age_oldest_frames": int((fr - births.min()).item()),
               # P225: the novelty rule's rate WITH its denominator
               "admission": {"checked": int(self.checked.item()),
                             "judged_new": int(self.checked_novel.item()),
                             "share_new": (round(float(self.checked_novel.item()) / float(self.checked.item()), 4)
                                           if int(self.checked.item()) else None)},
               "admitted": {"single_sense": int(self.admit_sense.item()),
                            "new_combination": int(self.admit_combo.item())},
               # capacity DEMAND while full: distinct turned-away patterns and
               # how often each was seen again (sizes the next table)
               "shadow": {"distinct": int(self.shadow_filled.item()),
                          "sightings": int(self.shadow_hits.sum().item()),
                          "overflow": int(self.shadow_overflow.item())}}
        # WHERE TODAY'S MOMENTS SIT (audit P238): the mean cosine between a moment and its nearest
        # slot (1 = the same pattern, 0 = nothing in common with any slot), and the share of
        # moments that belong to a slot under the admission rule's own line and therefore move its
        # centre (0 = no centre is moving: the table is a record of other moments; 1 = every
        # moment refines a slot). Running averages over about 1,000 frames, bias-corrected by
        # their own count, None until 100.
        _n = int(self._live_gauges_n)
        _g = (None if self._live_gauges is None or _n < 100
              else (self._live_gauges / (1.0 - 0.999 ** _n)).tolist())
        out["best_slot_cos"] = None if _g is None else round(_g[0], 4)
        out["moments_with_a_slot"] = None if _g is None else round(_g[1], 4)
        # DIFFERENT STORED PATTERNS (audit P238): slots_filled counts slots, not kinds of moment
        out.update(self._twin_census(k, occ))
        out["occupancy_effective_slots"] = round(math.exp(ent * math.log(max(k, 2))), 1)
        for tag in ("l2", "l3"):
            memb = self.membership(tag)[:k]
            out[f"{tag}_abs_mean"] = round(float(memb.abs().mean()), 6)
            out[f"{tag}_top"] = [round(float(v), 5)
                                 for v in torch.topk(memb, min(3, k)).values]
            # RANK-PERSISTENCE significance (2026-09-10, third design; corrected
            # 2026-09-18, audits P222 and P224): a slot counts only when its
            # correlation has OUT-RANKED its own eight sign-flip nulls persistently.
            # The line it must beat is judged per slot on that slot's OWN tick count,
            # is family-wise (the design's one-sided 3-sigma tail, alpha 0.00135,
            # divided by the number of slots judged together, so a whole table with
            # no link miscounts a slot 0.135 times in 100). The table's own empirical
            # null - the median standing plus the same z times 1.4826 x the median
            # absolute deviation (Efron) - is reported beside it, not applied (see the
            # measurement below).
            # Until the line is REACHABLE - first_evidence_tick(k), because the rank
            # EMA starts at 0.5 and moves a tenth of the way per tick - the count is
            # UNMEASURED, reported -1; a 0 before that is forced by arithmetic.
            ticks = int(getattr(self, f"{tag}_ticks").item())
            re = getattr(self, f"{tag}_rank_ema")[:k]
            window = max(1, int(round(1.0 / (1.0 - self.level_momentum[tag]))))
            need = int(self.first_evidence_tick(k))
            out[f"{tag}_ticks"] = ticks
            out[f"{tag}_rank_top"] = round(float(re.max()), 4)
            out[f"{tag}_rank_ceiling"] = round(self._rank_ceiling(ticks), 4)
            out[f"{tag}_ticks_needed"] = need
            out[f"{tag}_steps_to_first_evidence"] = max(0, (need - ticks) * window)
            # HOW MUCH OF THE TABLE CAN BE COUNTED (audit P239): the line is judged on each
            # slot's OWN ticks, so a slot minted after the table's ticks began adds a 0 that
            # is forced by arithmetic until its own count reaches `need`.
            _own = getattr(self, f"{tag}_slot_ticks")[:k]
            out[f"{tag}_slots_eligible"] = int((_own >= need).sum().item())
            out[f"{tag}_slot_ticks_min"] = int(_own.min().item())
            out[f"{tag}_slot_ticks_median"] = int(_own.median().item())
            if ticks >= need:
                st = getattr(self, f"{tag}_slot_ticks")[:k].clamp_min(1)
                neff = st.float().clamp(max=1.0 / (1.0 - self.rank_momentum))
                sigma = math.sqrt(1.0 / 12.0) / neff.sqrt()
                z_k = float(torch.special.ndtri(torch.tensor(1.0 - self.RANK_ALPHA / k, dtype=torch.float64)))
                line_ex = 0.5 + z_k * sigma
                mature = st >= 3
                if int(mature.sum()) >= 8:
                    centre = float(re[mature].median())
                    spread = 1.4826 * float((re[mature] - centre).abs().median())
                else:
                    centre, spread = 0.5, 0.0
                line_emp = centre + z_k * spread
                # MEASURED 2026-09-18 (vf_p224_fix.py): applying max(line_ex, line_emp)
                # as the gate gave 0 false alarms in 384 no-link slots but counted
                # NOTHING on the planted-effect fixture in 6 of 6 seeds - the planted
                # effect lifts the table's middle and the floor read 0.95-1.08. The
                # family-wise line alone gave 0 of 384 false alarms and 6 of 6 detections,
                # so it is the gate; the table's middle and spread are REPORTED beside
                # the count (rank_mean, rank_sd, line_empirical), never applied.
                line = line_ex
                legacy = self._rank_line(ticks)
                out[f"{tag}_line"] = round(float(line.min()), 4)
                out[f"{tag}_line_empirical"] = round(float(line_emp), 4)
                out[f"{tag}_sig"] = int(((re > line) & mature).sum().item())
                out[f"{tag}_line_1slot"] = round(legacy, 4)
                out[f"{tag}_sig_1slot"] = int((re > legacy).sum().item())
                out[f"{tag}_expected_false_1slot"] = round(k * self.RANK_ALPHA, 2)
                out[f"{tag}_expected_false"] = self.RANK_ALPHA
                out[f"{tag}_rank_mean"] = round(float(re[mature].mean()), 4) if int(mature.sum()) else None
                out[f"{tag}_rank_sd"] = round(float(re[mature].std()), 4) if int(mature.sum()) > 1 else None
            else:
                out[f"{tag}_line"] = None
                out[f"{tag}_sig"] = -1     # unmeasured: the line is not yet reachable
        return out
