# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Section 24: fast transients, measured at their own timescale, never averaged away.

A slip, a sudden loss of support or a 100 ms destabilisation must not disappear inside a long
average. Two lanes, each summarised with the addendum's five measures:

  world lane  per physics frame (10 ms on the crib): head accelerometer deviation from gravity,
              head gyro, total force from outside the body (a fall below its usual level is a
              loss of support: support_drop below), the largest single contact force (an impact)
  learner lane per model step (50 ms): each body's next-moment prediction error per sense

  instantaneous       the value at its own resolution
  step change         the change from the previous sample
  short peak          the largest value within PEAK_WINDOW (100 ms)
  area of the spike   the excess over the running median, summed while the spike lasts (unit x s)
  recovery            the time from the peak until the series is back within RECOVER_FACTOR of
                      its running median

The three timescales the addendum names: 50-100 ms (this module), 250 ms - 2 s (the detectors'
windows) and longer (the tracker's windows).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

G = 9.81
FRAME_S = 0.01
STEP_S = 0.05
PEAK_WINDOW_S = 0.10
SPIKE_MADS = 6.0                 # a spike: above the running median by 6 median absolute deviations
JOLT_FLOOR_MS2 = 2.0             # ...and, for the head's acceleration, by at least 2 m/s2: a SPIKE is
                                 # sudden against its own recent level; primitives.JOLT_MS2 (2 g) is the
                                 # separate, absolute line of a head jolt event
RECOVER_FACTOR = 1.5
MEDIAN_WINDOW = 200              # samples in the running median / MAD


def spikes(x: np.ndarray, dt: float, *, min_excess: float = 0.0) -> List[dict]:
    """Spikes of a 1-D series: start, peak index, peak value, excess area, recovery seconds."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = []
    if n < 5:
        return out
    fin = np.isfinite(x)
    med = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - MEDIAN_WINDOW)
        seg = x[a:i + 1][fin[a:i + 1]]
        if seg.size >= 5:
            med[i] = np.median(seg)
            mad[i] = np.median(np.abs(seg - med[i])) + 1e-12
    i = 0
    while i < n:
        if fin[i] and np.isfinite(med[i]) and x[i] - med[i] > max(SPIKE_MADS * mad[i], min_excess):
            j = i
            peak = i
            base = med[i]
            area = 0.0
            while j < n and fin[j] and x[j] > base * RECOVER_FACTOR + 1e-12 and x[j] - base > 0:
                if x[j] > x[peak]:
                    peak = j
                area += (x[j] - base) * dt
                j += 1
            out.append(dict(start=i, peak=int(peak), value=float(x[peak]), baseline=float(base),
                            excess=float(x[peak] - base), area=float(area),
                            recovery_s=float((j - peak) * dt), recovered=bool(j < n)))
            i = max(j, i + 1)
        else:
            i += 1
    return out


def running_median(x: np.ndarray, window: int = MEDIAN_WINDOW, min_count: int = 5) -> np.ndarray:
    """The median of each sample and the `window` samples before it (NaN ignored; NaN where fewer
    than min_count finite samples): the series' usual level, as spikes() reads it."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    w = int(window) + 1
    padded = np.r_[np.full(w - 1, np.nan), x]
    view = np.lib.stride_tricks.sliding_window_view(padded, w)          # row i: x[i-window..i]
    enough = np.sum(np.isfinite(view), axis=1) >= min_count
    if enough.any():
        with np.errstate(all="ignore"):
            out[enough] = np.nanmedian(view[enough], axis=1)
    return out


SUPPORT_HOLD_FRAMES = 3          # a fall must hold 3 frames (30 ms): a one-frame swing of the
                                 # contact solver is no loss of support


def support_fall(of: np.ndarray) -> np.ndarray:
    """How far the force from outside the body falls BELOW ITS USUAL LEVEL and stays there: the
    running median of the previous 2 s (MEDIAN_WINDOW frames) minus the most of the next three
    frames, never negative; placed at the first low frame.

    Until 2026-09-24 (detector v5) the series was the force AT a frame minus the most of the next
    three, so the falling edge of a one-frame impact spike counted as a loss of support although
    the force only returned to its usual level. Measured 2026-09-24 through summarize_world, per
    10,000 frames of 10 ms (with the frame a fall starts from judged against this running median):
    production (179,000 frames, segments 38 and 39) 704 falls under v5, 10,658 of its 12,610 falls
    (85%) starting a quarter body weight ABOVE the usual level, and 95 under this series;
    random-order babble (96,000 frames) 799 and 90; a limp body (96,000) 0 and 0. (The first
    measurement, OBSERVER_DESIGN.md section 9: 110, 97.5 and 0 on 32,000 frames each.)"""
    of = np.asarray(of, dtype=np.float64)
    h = SUPPORT_HOLD_FRAMES
    out = np.zeros(of.size)
    if of.size <= h:
        return out
    usual = running_median(of)
    ahead = np.max(np.stack([of[1 + k:of.size - h + 1 + k] for k in range(h)]), axis=0)
    fall = usual[:-h] - ahead
    out[1:of.size - h + 1] = np.where(np.isfinite(fall), np.maximum(0.0, fall), 0.0)
    return out


def world_series(fast: np.ndarray, frames: np.ndarray) -> Dict[str, np.ndarray]:
    """Frame-resolution series from a track's fast lane [N, 5, 8] (unused slots NaN)."""
    out = {}
    names = ("head_jolt", "head_spin", "outside_force", "max_contact_force", "head_force")
    for k, nm in enumerate(names):
        rows = []
        for i in range(len(fast)):
            v = fast[i, k, :int(frames[i])]
            rows.append(np.abs(v - G) if nm == "head_jolt" else v)
        out[nm] = np.concatenate(rows) if rows else np.zeros(0)
    out["support_drop"] = support_fall(out["outside_force"])
    return out


def world_floors(weight_n: float):
    """(lane, the least excess over its running median that counts as a spike) for the world lanes."""
    return (("head_jolt", JOLT_FLOOR_MS2), ("head_spin", 1.0), ("support_drop", 0.25 * weight_n),
            ("max_contact_force", 0.25 * weight_n))


def binned(x: np.ndarray, dt: float, rec: np.ndarray, keep: np.ndarray, bins: np.ndarray, *,
           min_excess: float = 0.0, ratio: bool = False) -> Dict[int, dict]:
    """One lane's summary per bin, from ONE continuous series (analysis v5, 2026-09-25): the spikes of
    x are found over the whole series - a digestion window's track with its look-back and its
    held-back end - and each is counted in the bin of its PEAK's record, only when that record is
    kept (finalized by this window). rec[i] is sample i's record, keep[r] whether record r is
    counted here, bins[r] its bin. Per bin: samples (finite samples of kept records), spikes,
    largest (the largest spike peak, 0 without one), area_sum, recovery_sum_s, short_peak (the
    largest kept sample); with ratio, largest_ratio (the largest spike peak, or the median without
    one, over the median of the bin's kept samples: the learner lane's).

    Until v5 each bin of 1,000 steps was summarized from its own samples inside one window, so a
    spike's running median restarted at every bin edge and every window edge, and a bin could not
    be cut at a lineage cut inside it (daemon.cut_fast): the bins are now the exposure's 100 steps,
    which only a whole series makes possible without an edge every 500 frames."""
    x = np.asarray(x, dtype=np.float64)
    rec = np.asarray(rec, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    bins = np.asarray(bins, dtype=np.int64)
    out: Dict[int, dict] = {}
    if not x.size or not keep.any():
        return out
    kept = keep[rec]
    sb = bins[rec]
    for b in np.unique(bins[keep]):
        v = x[kept & (sb == b)]
        fin = v[np.isfinite(v)]
        out[int(b)] = dict(samples=int(fin.size), spikes=0, largest=0.0, area_sum=0.0, recovery_sum_s=0.0,
                           short_peak=float(fin.max()) if fin.size else None)
        if ratio:
            out[int(b)]["_median"] = float(np.median(fin)) if fin.size else None
    for p in spikes(x, dt, min_excess=min_excess):
        r = int(rec[p["peak"]])
        if not keep[r]:
            continue                      # its peak is another window's (or not finalized yet)
        o = out[int(bins[r])]
        o["spikes"] += 1
        o["largest"] = max(o["largest"], float(p["value"]))
        o["area_sum"] += float(p["area"])
        o["recovery_sum_s"] += float(p["recovery_s"])
    if ratio:
        for o in out.values():
            med = o.pop("_median")
            if med is not None:
                o["largest_ratio"] = float((o["largest"] if o["spikes"] else med) / max(med, 1e-12))
    return out


def summarize_world(fast: np.ndarray, frames: np.ndarray, weight_n: float) -> dict:
    s = world_series(fast, frames)
    res = {}
    for nm, floor in world_floors(weight_n):
        x = s[nm]
        sp = spikes(x, FRAME_S, min_excess=floor)
        res[nm] = dict(samples=int(x.size), spikes=len(sp),
                       largest=float(max((p["value"] for p in sp), default=0.0)),
                       area_sum=float(sum(p["area"] for p in sp)),
                       recovery_sum_s=float(sum(p["recovery_s"] for p in sp)),
                       short_peak=float(np.nanmax(x)) if x.size else None)
    return res


def summarize_learner(series: np.ndarray) -> dict:
    """One body's next-moment error for one sense, per model step."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return dict(samples=int(x.size), spikes=0)
    sp = spikes(x, STEP_S)
    med = float(np.median(x))
    return dict(samples=int(x.size), median=med, spikes=len(sp),
                largest=float(max((p["value"] for p in sp), default=0.0)),
                largest_ratio=float(max((p["value"] for p in sp), default=med) / max(med, 1e-12)),
                area_sum=float(sum(p["area"] for p in sp)),
                recovery_sum_s=float(sum(p["recovery_s"] for p in sp)),
                mean_step_change=float(np.mean(np.abs(np.diff(x)))))


def brief_vs_gradual(brief: np.ndarray, gradual: np.ndarray, slow_window: int) -> dict:
    """The harness comparison (section 24): the same total perturbation delivered in one brief
    pulse or spread out. The fast lane's short peak must separate them even when a slow average
    over slow_window samples does not."""
    def slow(x):
        x = np.asarray(x, dtype=np.float64)
        k = max(1, int(slow_window))
        return float(np.max(np.convolve(np.nan_to_num(x), np.ones(k) / k, mode="valid"))) if x.size >= k else float(np.nanmean(x))
    def peak(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanmax(x)) if x.size else 0.0
    pb, pg = peak(brief), peak(gradual)
    sb, sg = slow(brief), slow(gradual)
    return dict(peak_brief=pb, peak_gradual=pg, peak_ratio=pb / max(pg, 1e-12),
                slow_brief=sb, slow_gradual=sg, slow_ratio=sb / max(sg, 1e-12))
