# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The machine block of the stats record: memory, card memory and step time, every window.

WHY (docs/gain_of_function_v3/WORKLOG.md 17:20). H84 segment 31 slowed from 1.225 to 0.509
model steps per second between the records at 14:15 and 14:19 on 2026-09-22 and was stopped
three hours later; every phase of the step slowed, and nothing in the run's own record said how
much memory the machine had left, how close Windows' commit charge was to its limit, how much of
the graphics card other programs were holding, or how often this process was paging. This block
records exactly those, so the next slowdown carries its own reading.

Fields (all None when the platform cannot say):
  ram_available_gb   physical memory still free for new work, GB (0 = none left)
  commit_gb, commit_limit_gb
                     Windows commit charge and its limit, GB. When commit reaches the limit
                     Windows refuses the NEXT process that asks for memory (on 2026-09-21 that
                     was the training supervisor). Read with the same call as
                     mimo_bridge.tools.commit_guard.
  rss_gb             this trainer process's resident memory, GB
  gpu_used_mib, gpu_total_mib
                     graphics-card memory in use by ALL processes, and the card's total, MiB,
                     from nvidia-smi. Sampled once every `gpu_every` windows (default 10, about
                     one call per 13 minutes at 1.2 steps/s) because one call costs 100-300 ms;
                     gpu_reading_age_windows says how many windows old the reading is (0 = this
                     window). torch's own peak allocation cannot show another program's memory
                     or a spill into shared system memory; this can.
  torch_reserved_mib this process's CUDA memory held by torch's caching allocator, MiB
  page_faults_per_s  this process's page faults per wall second since the previous window
                     (Windows counts soft and hard faults together)
  step_ms, model_ms, env_ms, causal_chain_ms
                     wall milliseconds per model step in this window: the model's step
                     (encode, forward, backward, planning), the bodies' burst, the causal-chain
                     recorder's latch predictions (only with --causal-chain on) and their sum
  wellness_channels_ms, wellness_rank_ms_per_plan
                     (only when a wellness-channel flag is on) the channels' own code
                     (lem/model/wellness_channels.py: learning progress, controllability, the
                     ring's channel record, the ranking), wall milliseconds per model step -
                     a PART of model_ms, so a flags-on run and a flags-off run can be compared
                     step for step - and per ranking of the goal ring
  efference_ms       (only with --efference-heads on) the efference lesson's own forward passes
                     and gauges, milliseconds per timed step; part of model_ms (review of b1,
                     MINOR 6: the lesson had no timing of its own)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional


def _commit_reading():
    """(committed_gb, limit_gb, free_ram_gb) or None. The commit guard owns the call."""
    try:
        from mimo_bridge.tools.commit_guard import commit_reading
    except ImportError:
        return None
    try:
        return commit_reading()
    except OSError:
        return None


def nvidia_smi_memory(timeout_s: float = 5.0):
    """(used_mib, total_mib) of the first card, all processes, or None."""
    try:
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000      # CREATE_NO_WINDOW: no console flash
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=timeout_s, **kw)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        used, total = (float(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])
    except ValueError:
        return None
    return used, total


class MachineSampler:
    """One sample per stats window. gpu_every: sample the card once every N windows."""

    def __init__(self, gpu_every: int = 10, gpu_reader=nvidia_smi_memory, clock=time.time):
        self.gpu_every = max(1, int(gpu_every))
        self._gpu_reader = gpu_reader
        self._clock = clock
        self._windows = 0
        self._gpu = None
        self._gpu_window = None
        self._pf_last = None
        self._pf_time = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
            self._psutil = psutil
        except ImportError:                # the venv carries psutil; this keeps tests portable
            self._proc = None
            self._psutil = None

    def sample(self, model_ms: Optional[float], env_ms: Optional[float],
               causal_ms: Optional[float] = None, wellness: Optional[dict] = None, efference_ms: Optional[float] = None) -> dict:
        """causal_ms: the causal-chain recorder's own time per step (None when it is off);
        it is part of step_ms and reported beside it as causal_chain_ms. wellness: the model's
        wellness_channels_cost_window() (None when no wellness-channel flag is on). Efference timing is also part of model_ms."""
        gb = 1024.0 ** 3
        out = {"ram_available_gb": None, "commit_gb": None, "commit_limit_gb": None,
               "rss_gb": None, "gpu_used_mib": None, "gpu_total_mib": None,
               "gpu_reading_age_windows": None, "torch_reserved_mib": None,
               "page_faults_per_s": None,
               "model_ms": None if model_ms is None else round(float(model_ms), 1),
               "env_ms": None if env_ms is None else round(float(env_ms), 1),
               "step_ms": (None if model_ms is None or env_ms is None
                           else round(float(model_ms) + float(env_ms)
                                      + float(causal_ms or 0.0), 1))}
        if causal_ms is not None:
            out["causal_chain_ms"] = round(float(causal_ms), 1)
        if wellness:
            _n = int(wellness.get("steps") or 0)
            _p = int(wellness.get("planning_events") or 0)
            _tot = (float(wellness.get("per_step_ms") or 0.0) * _n
                    + float(wellness.get("planning_ms") or 0.0) * _p)
            out["wellness_channels_ms"] = round(_tot / _n, 2) if _n else None
            out["wellness_rank_ms_per_plan"] = (None if wellness.get("planning_ms") is None
                                                else round(float(wellness["planning_ms"]), 2))
        if efference_ms is not None:
            out["efference_ms"] = round(float(efference_ms), 2)
        if self._psutil is not None:
            try:
                out["ram_available_gb"] = round(self._psutil.virtual_memory().available / gb, 3)
                mi = self._proc.memory_info()
                out["rss_gb"] = round(mi.rss / gb, 3)
                pf = getattr(mi, "num_page_faults", None)
                now = self._clock()
                if pf is not None:
                    if self._pf_last is not None and now > self._pf_time:
                        out["page_faults_per_s"] = round((pf - self._pf_last) / (now - self._pf_time), 1)
                    self._pf_last, self._pf_time = pf, now
            except (OSError, AttributeError):
                pass
        c = _commit_reading()
        if c is not None:
            out["commit_gb"] = round(c[0], 3)
            out["commit_limit_gb"] = round(c[1], 3)
        if self._windows % self.gpu_every == 0:
            g = self._gpu_reader()
            if g is not None:
                self._gpu, self._gpu_window = g, self._windows
        if self._gpu is not None:
            out["gpu_used_mib"] = round(self._gpu[0], 1)
            out["gpu_total_mib"] = round(self._gpu[1], 1)
            out["gpu_reading_age_windows"] = int(self._windows - self._gpu_window)
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                out["torch_reserved_mib"] = round(torch.cuda.memory_reserved() / 2 ** 20, 1)
        except (ImportError, RuntimeError):
            pass
        self._windows += 1
        return out
