# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""strike_attribution.reading: the pure attribution rule.

Planner-driven needs BOTH clauses: the chosen orders alone keep at least half
the full strike rate, and the full rate clears twice the babble-only floor.
A rate that collapses without babble is babble-driven; the space between the
two rules refuses to decide (one seed is not power).
"""
from mimo_bridge.tools.strike_attribution import reading


def test_planner_driven_when_chosen_orders_carry_the_rate():
    # the 2026-09-04 measurement: 0.0567 / 0.0297 / 0.0093
    assert reading(0.0567, 0.0297, 0.0093) == "planner-driven"


def test_babble_driven_when_removing_babble_collapses_it():
    assert reading(0.05, 0.005, 0.001) == "babble-driven"


def test_refuses_when_normal_rate_sits_at_the_floor():
    # chosen orders keep their share, but the whole effect is floor-sized:
    # clause two fails and the rule must not claim the planner
    assert reading(0.010, 0.008, 0.009) == "mixed-or-underpowered"


def test_refuses_the_in_between_case():
    assert reading(0.05, 0.018, 0.001) == "mixed-or-underpowered"
