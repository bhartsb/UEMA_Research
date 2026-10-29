# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What the body map's handle statistic reads on inputs whose relation is known (2026-10-03).

The coherence wellness channel carries the body map's extent (lem/model/body_schema.py self_extent), and the
movement drive's bonus is multiplied by the handles' mean number (movement.self_factor). The pages called that
number the share of the body the map "calls its own" and the share of movement that is "self-caused". The
statistic is, per handle, a running average of (size of the command's change x size of each proprioception
number's change), each size divided by its own running root-mean-square, passed through
sigmoid(5 x (mean - one half)). It has no time-lag bank, no sign and no comparison against movement without a
command, so it counts co-movement and not causation (operator's decision of Oct 3, 2026: named co-movement on
the pages now, replaced after the motor-encoder repair; docs/gain_of_function_v4/coordinator/
own_command_analysis_2026-10-02.html).

These tests feed the production class synthetic commands and proprioception and hold the readings the pages
quote: two unrelated random signals read well above the scale's low end (30 of 100 when the command changes on
1 model step in 5, 80 of 100 when it changes on every model step), and a body that answers every command one
model step late reads the low end, the same as a body that is moved while its command never changes. CPU only,
26 handles, 6,000 model steps per case, no model and no world.
"""
import math

import torch

from lem.model.body_schema import DifferentiableBodySchema

H, P, M, STEPS = 26, 11, 9, 6000
FLOOR = 1.0 / (1.0 + math.exp(2.5))      # sigmoid(5 x (0 - one half)): a score of 0
TOP = 1.0 / (1.0 + math.exp(-2.5))       # sigmoid(5 x (1 - one half)): a score of 1


def _reading(make):
    """The handles' mean co-movement number after STEPS model steps of make(t, command, proprioception)."""
    torch.manual_seed(0)
    schema = DifferentiableBodySchema(n_handles=H)
    command, proprio = torch.zeros(H, M), torch.zeros(H, 12)
    for t in range(STEPS):
        command, proprio = make(t, command, proprio)
        schema.update(command, proprio)
    return float(schema.self_membership().mean())


def _moved(proprio, change):
    """proprio with `change` added to its first 11 columns (the 12th, the row's index, is not read)."""
    return torch.cat([proprio[:, :P] + change, proprio[:, P:]], dim=1)


def test_unrelated_changes_do_not_read_the_low_end():
    """Two random signals with no relation: the product of two change sizes averages above zero, so the reading
    sits far above the floor, and where it sits depends on how often the command changes."""
    every = _reading(lambda t, c, p: (c + torch.randn(H, M), _moved(p, torch.randn(H, P))))
    hold5 = _reading(lambda t, c, p: (c + torch.randn(H, M) if t % 5 == 0 else c, _moved(p, torch.randn(H, P))))
    assert 0.77 < every < 0.83, every          # 80 of 100, above the middle of the range
    assert 0.27 < hold5 < 0.33, hold5          # 30 of 100; H84 read 28 of 100 at model step 455,800
    assert every > 0.5 > hold5 > FLOOR + 0.15


def test_a_response_one_model_step_late_reads_the_low_end():
    """Every proprioception number changes by exactly its own handle's command-change size, one model step after
    the command changed (the command changes on 1 model step in 5): fully caused, never in the same model step,
    so every product is 0 and the reading is the floor."""
    pending = {"size": torch.zeros(H, 1)}

    def late(t, command, proprio):
        new = command + torch.randn(H, M) if t % 5 == 0 else command
        change = torch.sign(torch.randn(H, P)) * pending["size"]
        pending["size"] = (new - command).norm(dim=-1, keepdim=True)
        return new, _moved(proprio, change)

    assert abs(_reading(late) - FLOOR) < 1e-3


def test_movement_without_any_command_change_reads_the_low_end():
    """The body is moved while its command never changes: the same reading as the late response above, so the
    statistic cannot tell the two apart."""
    assert abs(_reading(lambda t, c, p: (c, _moved(p, torch.randn(H, P)))) - FLOOR) < 1e-3


def test_equal_change_sizes_in_the_same_model_step_read_near_the_top():
    """Every proprioception number of a handle changes by that handle's command-change size in the same model
    step. A handle's score also pairs its command with the other 25 handles' proprioception, so it stays under 1."""

    def together(t, command, proprio):
        d = torch.randn(H, M)
        return command + d, _moved(proprio, torch.sign(torch.randn(H, P)) * d.norm(dim=-1, keepdim=True))

    reading = _reading(together)
    assert 0.88 < reading < TOP, reading


def test_the_pages_name_the_statistic_co_movement_and_quote_these_readings():
    """The explainer's coherence entry, the dashboards' rows and the lineage series say what the statistic is and
    that it is not causation; the two old claims are gone."""
    from mimo_bridge.explainer import schema as SCH
    from mimo_bridge.tools import dashboard as LEGACY
    from mimo_bridge.tools import lineage_series as LS

    entry = SCH.DOC["term.eval-coherence"]
    assert "co-movement" in entry["t"].lower()
    for words in ("no time-lag bank", "no sign", "passive-movement comparison", "80 of 100", "30 of 100",
                  "8 of 100", "sigmoid(5 x (score - one half))"):
        assert words in entry["h"], words
    assert "co-movement" in SCH.DOC["incorporation"]["t"]
    assert "co-movement" in SCH.DOC["term.eval-movement"]["w"]
    for text in (entry["w"], SCH.DOC["incorporation"]["w"], SCH.DOC["term.eval-movement"]["w"]):
        assert "body's own doing" not in text and "self-caused" not in text
    assert "Co-movement channel (body map)" in LEGACY.ROW_DOC and "Self-causation channel" not in LEGACY.ROW_DOC
    assert "Body map co-movement (incorporation)" in LEGACY.ROW_DOC
    said = LEGACY.self_causation_state([0.27, 0.29])
    assert said.startswith("measured") and "co-movement" in said and "self-caused" not in said
    assert "co-movement" in LS.SERIES["extent"]["title"] and "co-movement" in LS.SERIES["wellness.coherence"]["title"]
    assert "does not measure whether the command caused" in LS.SERIES["wellness.coherence"]["note"]
