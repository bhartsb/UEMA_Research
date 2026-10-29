# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A probe detects a planning event without wiping the record the persistent memory reads (2026-09-29).

memory_integration.observe reads the last chosen goal from model._last_plan_inputs every step and production never
clears it; probes that set it to None before an observe fed the memory "no goal", and the remembered-hand comparison
failed its own repeat check at 414,200 on that.
"""
import re
from pathlib import Path
from types import SimpleNamespace

from mimo_bridge.tools.gof_probe_harness import fresh_plan_inputs

REPO = Path(__file__).resolve().parents[4]


def test_a_new_record_is_this_observes_plan_and_the_same_record_is_none():
    model = SimpleNamespace(_last_plan_inputs=None)
    prior = model._last_plan_inputs
    assert fresh_plan_inputs(model, prior) is None
    model._last_plan_inputs = {"goal": 1}
    assert fresh_plan_inputs(model, prior) == {"goal": 1}
    prior = model._last_plan_inputs
    assert fresh_plan_inputs(model, prior) is None          # a step that did not plan keeps the old record
    model._last_plan_inputs = {"goal": 1}                    # an equal but new record is a new plan
    assert fresh_plan_inputs(model, prior) is not None


def test_no_probe_or_harness_wipes_the_record_the_memory_reads():
    wipe = re.compile(r"_last_plan_inputs\s*=\s*None")
    offenders = []
    for folder in ("Source/Python/mimo_bridge/tools", "docs/gain_of_function_v3", "docs/gain_of_function_v4"):
        for path in (REPO / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                if wipe.search(line) and not line.lstrip().startswith("#"):
                    offenders.append("%s:%d" % (path.relative_to(REPO), n))
    assert not offenders, "use gof_probe_harness.fresh_plan_inputs instead: %s" % offenders
