# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The model alert's verdict on a flagged layer covers only as many idle units as its measurement switched off."""
from mimo_bridge.tools.development_dashboard import LAYER_VERDICTS, layer_verdict


def test_verdict_holds_only_up_to_the_units_its_test_covered():
    for (blk, name), known in LAYER_VERDICTS.items():
        covered = max(known['idle'], known.get('covered', known['idle']))
        at_measured = 1 - covered / known['width']
        assert layer_verdict(blk, name, at_measured).startswith('No action needed')
        more_idle = 1 - (covered + 3) / known['width']
        assert layer_verdict(blk, name, more_idle).startswith('Needs a new check')


def test_an_unmeasured_layer_says_so():
    assert layer_verdict('model', 'some_new_layer', 0.3).startswith('Not yet checked')
