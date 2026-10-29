# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Physical episode boundaries: explicit schedule or a measured world fault."""
import math
from collections import deque


def physical_fault(frames, previous_times=None):
    if not frames: return 'empty observation batch'
    now=[float(f.sim_time) for f in frames]
    if not all(math.isfinite(t) for t in now): return 'nonfinite simulation time'
    for i,f in enumerate(frames):
        if (f.meta or {}).get('physical_state_finite') is False:
            return f'body {i}: nonfinite physical state or skin measurement'
    if previous_times is not None:
        if len(previous_times) != len(now): return 'body batch changed'
        if any(t < p for t,p in zip(now,previous_times)): return 'physics clock moved backwards'
        if any(t == p for t,p in zip(now,previous_times)): return 'physics clock did not advance'
    return None


def scheduled_reset(step, episode_steps):
    if episode_steps < 0: raise ValueError('episode_steps cannot be negative')
    return episode_steps > 0 and step > 0 and step % episode_steps == 0


class PhysicalFaultBudget:
    """Three faults within 100 simulated seconds require repair, even with good steps between."""
    def __init__(self, window_steps=2000, maximum=3):
        self.window_steps=window_steps;self.maximum=maximum;self.failures=deque()

    def record(self,step):
        while self.failures and step-self.failures[0]>self.window_steps: self.failures.popleft()
        self.failures.append(step)
        return len(self.failures)>=self.maximum
