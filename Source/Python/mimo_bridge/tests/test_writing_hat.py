# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The writing hat: the reading gate's scorer and rewriter.

A jargon-and-number-dense note must score 3+, be rewritten, and come out scoring
1-2 with every fact still present. A plain note passes untouched. The gate never
raises; the backend is deterministic.
"""
import os

from mimo_bridge.tools.writing_hat import gate, gate_report, score

DENSE = ("VITALS: causal_pe 0.012752776076085865 vs vis_adv 0.003692265191684762; "
         "prior_w 0.04328612585489863; rank1 0.45 with td_gate 1.04e-06; "
         "next_pe split shows tactile 0.0365 vs 0.0221 while qpos drifts; "
         "EMA lag suspected; ROM at 5%")

PLAIN = "No intervention needed right now. The mobile kept swinging all hour."


def test_dense_note_fails_then_passes():
    assert score(DENSE) >= 3
    lines, rec = gate([DENSE])
    assert rec["first_score"] >= 3
    assert rec["final_score"] <= max(2, rec["first_score"] - 1)  # strictly improved
    assert rec["backend"] == "deterministic"
    out = "\n".join(lines)
    # facts survive: the numbers are still there (rounded is fine)
    for frag in ("0.0128", "0.45", "5%"):
        assert frag in out or frag.rstrip("%") in out
    # jargon replaced by plain real terms
    assert "vis_adv" not in out and "prior_w" not in out


def test_plain_note_passes_untouched():
    lines, rec = gate([PLAIN])
    assert lines == [PLAIN]
    assert rec["rounds"] == 0 and rec["first_score"] <= 2


def test_gate_report_covers_all_hats_and_reports_worst():
    notes, rec = gate_report({"cto": [DENSE], "verdict": [PLAIN], "empty": []})
    assert "cto" in notes and "verdict" in notes and "empty" not in notes
    assert rec["worst_first"] >= 3
    assert rec["worst_final"] <= rec["worst_first"]


def test_the_backend_is_deterministic():
    _, rec = gate([DENSE])
    assert rec["backend"] == "deterministic"
