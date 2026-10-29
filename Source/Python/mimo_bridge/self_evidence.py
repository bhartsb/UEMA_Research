# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The evidence test the self-model's slow levels share (UEMA spec 11.4).

One place for the arithmetic that decides when a link between something's PRESENCE and the
body's WELLNESS has been shown, rather than drawn by luck. Both users of it - the
hierarchical self in hierarchical_self.py and the entity self in entity_self.py -
call these functions, so the two can never drift apart.

The test, in plain terms. Each level keeps a score for every candidate, from 0 to 1: the
share of its own fake comparison signals (controls) whose link it beat, averaged over the
windows it has lived, with a long memory. A candidate with no true link centres on 0.5.
A candidate passes when that score clears a PASS MARK. Because many candidates are judged
at once, luck gets many tries, so the pass mark rises with the number judged: the one-sided
tail ALPHA is divided by that number (the Bonferroni correction), which is what
`rank_line(..., k=...)` does.

Nothing here holds state or touches the graph; every function is a plain calculation.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

# The design's own one-sided 3-sigma tail: the share of judgements allowed to be luck.
RANK_ALPHA = 0.00135
# How long the score remembers: 0.9 keeps about 10 windows.
RANK_MOMENTUM = 0.9


def rank_sigma(ticks, rank_momentum: float = RANK_MOMENTUM) -> float:
    """How much the score of a candidate with NO true link wanders, after this many windows.

    One window's rank is uniform on 0 to 1 under no link, whose variance is 1/12. The score
    is an average with a memory of 1 / (1 - momentum) windows, so its spread shrinks as the
    square root of the windows lived, and stops shrinking once the memory is full."""
    neff = min(float(ticks), 1.0 / (1.0 - rank_momentum))
    return math.sqrt(1.0 / 12.0) / math.sqrt(max(neff, 1e-9))


def rank_line(ticks, k: Optional[int] = None, rank_momentum: float = RANK_MOMENTUM,
              alpha: float = RANK_ALPHA) -> float:
    """The pass mark after this many windows. k = how many candidates are judged together;
    None keeps the old one-at-a-time mark (0.5 + 3 sigma), which lets luck through once per
    k judgements instead of once per table."""
    if k is None:
        return 0.5 + 3.0 * rank_sigma(ticks, rank_momentum)
    z = float(torch.special.ndtri(torch.tensor(1.0 - alpha / max(int(k), 1),
                                               dtype=torch.float64)))
    return 0.5 + z * rank_sigma(ticks, rank_momentum)


def rank_ceiling(ticks, rank_momentum: float = RANK_MOMENTUM) -> float:
    """The highest score reachable after this many windows by a candidate that beats every
    control every time. The score starts at 0.5 and closes (1 - momentum) of the way to 1
    each window, so it approaches 1 but never arrives."""
    return 0.5 + 0.5 * (1.0 - rank_momentum ** int(ticks))


def first_evidence_tick(k: Optional[int] = None, rank_momentum: float = RANK_MOMENTUM,
                        alpha: float = RANK_ALPHA, limit: int = 100000) -> Optional[int]:
    """Windows a PERFECT candidate needs before it could pass: the first window at which the
    highest reachable score is above the pass mark. Before it, a count of 0 is forced by
    arithmetic and means nothing. None when the mark is out of reach at every window, which
    happens once k is so large that the mark sits above 1."""
    for n in range(1, limit):
        if rank_ceiling(n, rank_momentum) > rank_line(n, k, rank_momentum, alpha):
            return n
    return None


def first_evidence_steps(window: int, k: Optional[int] = None,
                         rank_momentum: float = RANK_MOMENTUM,
                         alpha: float = RANK_ALPHA) -> Optional[int]:
    """The same in moments: two windows for the controls to warm up, then the climb."""
    need = first_evidence_tick(k, rank_momentum, alpha)
    return None if need is None else (2 + need) * int(window)


def flip_signs(block: int, n_nulls: int) -> List[float]:
    """The control signs for one block of moments: +1 or -1 per control, from a hash of the
    block number and the control's own salt (a splitmix-style integer mix). Deterministic, so
    a resumed run continues the same schedule with nothing stored."""
    out: List[float] = []
    for salt in range(n_nulls):
        z = ((block + salt * 0x632BE59B) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
        z ^= z >> 16
        z = (z * 0x45D9F3B) & 0xFFFFFFFF
        z ^= z >> 16
        out.append(1.0 if (z & 1) else -1.0)
    return out


def bias_correction(momentum: float, updates) -> float:
    """What to divide a running average by, when it was started at zero.

    An average that starts at 0 and takes (1 - momentum) of each new value is short of the
    true average by momentum to the power of the updates so far. Dividing by
    1 - momentum**updates removes that. Without it a slow average sits below the signal for
    its whole first window, which makes every candidate's early link read positive: measured
    on run H84 at moment 138,812, the hours-level slow average of wellness read 2.52 against
    a wellness of 3.34, and all 6,400 candidates' links were positive."""
    n = int(updates)
    if n <= 0:
        return 1.0
    return max(1.0 - momentum ** n, 1e-12)
