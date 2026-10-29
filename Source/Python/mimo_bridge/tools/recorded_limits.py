# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Resolve copied physical limits from an explicit, hashed experiment record."""
import hashlib
import json
import math
from pathlib import Path

from .causal_test import _limits_for_run


def recorded_limits(checkpoint, model, declaration=None):
    if declaration is None:
        limits, source = _limits_for_run(str(checkpoint), model)
        return limits, dict(source=source)
    path = Path(declaration).resolve()
    data = path.read_bytes()
    limits = dict(json.loads(data)['limits'])
    from mimo_bridge.train_mimo import PHASE_AGE
    if limits.get('phase_name') not in PHASE_AGE:
        raise ValueError('Recorded developmental phase is unknown')
    for field in ('rom_limit', 'strength_limit'):
        value = float(limits[field])
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'Invalid recorded {field}')
        limits[field] = value
    if 'age_months' in limits:
        if float(limits['age_months']) != float(PHASE_AGE[limits['phase_name']]):
            raise ValueError('Recorded physical age disagrees with native phase mapping')
    if 'handle_mask' in limits:
        mask = limits['handle_mask']
        if not isinstance(mask, list) or not mask or any(type(x) is not bool for x in mask):
            raise ValueError('Recorded motor mask must be a nonempty list of booleans')
    return limits, dict(source=str(path), sha256=hashlib.sha256(data).hexdigest(),
                        scope='Explicit recorded physical regime; no inferred checkpoint age')
