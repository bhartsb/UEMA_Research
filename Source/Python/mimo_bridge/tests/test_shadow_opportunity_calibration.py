# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
from mimo_bridge.tools.shadow_opportunity_calibration import resolution_requirement

PROFILE=dict(minimum_bits=2,maximum_bits=23,pressure_range_pa=100000.)


def test_requirement_resolves_physical_contrast_and_matches_live_rounding():
    for contrast in (10.,100.,1000.,10000.):
        r=resolution_requirement(PROFILE,contrast)
        bits=round(2+21*np.float32(r['minimum_resolution']))
        assert bits>=r['required_bits']
        assert 2*100000/(2**bits-1)<contrast
    assert round(2+21*.266)==8
    assert 100000/(2**8-1)==pytest.approx(392.1568627)


def test_missing_too_fine_or_nonfinite_distinction_cannot_invent_viability():
    for separation in (0.,-1.,float('nan'),float('inf'),1e-12):
        with pytest.raises(ValueError):resolution_requirement(PROFILE,separation)
