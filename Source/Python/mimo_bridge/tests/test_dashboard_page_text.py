# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Every sentence the dashboards print passes DESIGN.md section 11 and the reviews' reader findings.

Written by fix-dashboards (2026-09-23) for docs/gain_of_function_v3/reviews/dashboards.md (B1, M1-M10,
m1-m9) and the reader findings of the b1, b3, b4 and probes reviews. The Now cards' sentences are
built in Python (lineage_series.gain_of_function -> item_text is exactly what the page prints), so
the text is rendered here from the producers' real summary() objects where they exist, from the
field contract of 2026-09-23 where the model lane is still adding a field, and at the edges (shares
above 1 and below 0, a probe value of 7.3e-8, a 16-digit distance). Each card must carry zero bare
score decimals, no raw 16-digit float, no sentence printed twice, and its control.
"""
import html as _html
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest
import torch

from mimo_bridge.tools import development_dashboard as dd
from mimo_bridge.tools import lineage_series as ls
from mimo_bridge.tools.check_no_bare_decimals import (bare_decimals, doubled_sentences, long_numbers, page_text,
                                                       readout_problems, unanchored_numbers)

sys.path.insert(0, os.path.dirname(__file__))


def clean_text(text):
    """(bare decimals, long numbers, doubled sentences) found in one card's text."""
    return bare_decimals(text), long_numbers(text), doubled_sentences(text)


def unplaceable(text):
    """The numbers a reader cannot place that the bare-decimal rule passes: a score of 1 or more with
    no unit, a multiplier with nothing named after it, a raw input value of a hidden unit (the
    readout scanner's rules, commit 22b047f9b; held on every page text since fix-readout-dashboards,
    2026-09-23)."""
    return unanchored_numbers(text)


# the style guide writing rules (added by verify-fix-dashboards 2026-09-23): the banned wording and the
# status shorthand that may not appear without its numbers, in anything a person reads
BANNED = re.compile(r"(?i)\b(honest\w*|truthful|barely|slightly|roughly|significantly|substantially|in band|"
                    r"healthy|armed|nominal|a bit)\b")


def banned_words(text):
    return sorted({m.group(0).lower() for m in BANNED.finditer(text)})


def assert_plain(text, where):
    b, l, d = clean_text(text)
    assert not b, f"{where}: bare score decimals {b[:3]}"
    assert not l, f"{where}: raw long floats {l[:3]}"
    assert not d, f"{where}: sentences printed twice {d[:2]}"
    assert not banned_words(text), f"{where}: banned wording {banned_words(text)}"
    u = unplaceable(text)
    assert not u, f"{where}: numbers a reader cannot place {u[:3]}"


# --------------------------------------------------------------------------- the producers
def controllability_summary(follow=1.0, B=2, H=26, gaze=23, orders=12, hold=5, seed=0):
    """lem/model/wellness_channels.ControllabilityTracker fed bodies that follow (follow=1) or
    ignore (follow=0) 20-degree orders, graded by the tracker itself."""
    from lem.model import wellness_channels as WC
    from mimo_bridge.tools.probes import _axis_angle_quat, _quat_mul
    g = torch.Generator().manual_seed(seed)
    tr = WC.ControllabilityTracker(H, gaze, hold, dict(WC.CONTROL_FLOOR))
    ach = torch.zeros(B, H, 10)
    ach[..., 6] = 1.0
    step = 0
    for k in range(orders):
        axis = torch.randn(B, H, 3, generator=g)
        exec_t = torch.zeros(B, H, 9)
        exec_t[..., 3:7] = _quat_mul(_axis_angle_quat(axis, 20.0), ach[..., 3:7])
        exec_t[..., 7:9] = 0.5
        start = ach[..., 3:7].clone()
        for t in range(hold):
            step += 1
            ach = ach.clone()
            ach[..., 3:7] = _quat_mul(_axis_angle_quat(axis, 20.0 * follow * (t + 1) / hold), start)
            tr.step(step, exec_t, k, ach, "planner", None)
    return tr.summary(B)


def lp_summary(B=2, steps=400, falling=True):
    from lem.model import wellness_channels as WC
    tr = WC.LearningProgressTracker()
    for s in range(steps):
        err = torch.full((B,), (1.0 - 0.5 * s / steps) if falling else 1.0)
        tr.update("vision", err, torch.ones(B))
        tr.update("proprioception", err * 1.1, torch.ones(B))
        tr.tick_window(B)
    return tr.summary(B)


def rank_block(goal_age=None):
    from lem.model import wellness_channels as WC
    names = ["integrity", "controllability", "learning_progress", "coherence", "prediction_error"]
    sh = WC.RankShares()
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        contrib = torch.randn(40, len(names), 2, generator=g)
        contrib[:, 0] *= 3.0
        gated = torch.zeros(40, 2, dtype=torch.bool)
        gated[0] = True
        sh.add(contrib, contrib.sum(1).masked_fill(gated, float("-inf")), gated, gated.all(0), names)
    out = {"mode": "normalized", "curiosity_mode": "additive",
           "weights": {"integrity": 2.0, "controllability": 1.0, "learning_progress": 0.75, "coherence": 0.5,
                       "prediction_error": 0.25}}
    out.update(sh.summary())
    if goal_age:
        out["goal_age"] = goal_age
    return out


def contact_summary(**contract):
    from lem.model.contact_head import ContactTracker
    out = ContactTracker().summary()
    out.update(contract)
    return out


def chain_summary(**extra):
    from mimo_bridge.causal_chain import CausalChainRecorder
    out = CausalChainRecorder(None, hold=5).window_summary()
    out.update(extra)
    return out


def audit(vis=0.0005, tac=0.0009, pro=-0.0005, pers=-0.05):
    return {"modalities": {m: {"improvement_over_shuffled": v, "improvement_over_persistence": pers,
                               "sensitivity": 0.01}
                           for m, v in (("vision", vis), ("tactile", tac), ("proprioception", pro))}}


def rich_record():
    """A record carrying every block, from the producers where they exist and the field contract of
    2026-09-23 where the model lane is adding the field."""
    return {
        "type": "mimo", "step": 400, "timestamp": 1_790_000_400,
        "prediction_audit": audit(),
        "causal_chain": chain_summary(chosen_best_share=0.15, alternatives_n=4, planner_events=10,
                                      chosen_tied_share=0.55, tie_band=0.016, gap_to_best_alt_mean=0.0017,
                                      candidate_spread_mean=0.005, plan_exec_distance_cm=1.4,
                                      pred_wellness_chosen=3.2, pred_wellness_best_alt=3.21,
                                      wellness_change_corr=0.1),
        "efference": {"body_cm_err_latch": 1.2, "body_cm_do_nothing_latch": 2.0, "body_cm_shuffled_latch": 1.9,
                      "gaze_deg_err": [1.0, 2.0, 3.0], "gaze_deg_do_nothing": [2.0, 4.0, 6.0]},
        "controllability": dict(controllability_summary(), difference_mean=0.99),
        "learning_progress": lp_summary(),
        "wellness_rank": rank_block({"mean_share_back": 0.93, "oldest_tenth_share": 0.95,
                                     "newest_tenth_share": 0.0, "events": 44}),
        "contact_control": contact_summary(onset_auc_self=0.737, control_onset_auc_self=0.824,
                                           onset_auc_world=0.6, control_onset_auc_world=0.7,
                                           success_paired=0.53, success_control=0.47, head_auc=0.919,
                                           gate_self_open=False, gate_world_open=True, habituation_mean=0.25,
                                           liveness={"trunk.0": 0.2, "patch.0": 0.9}),
        "synergy": {"dim": 32, "explained_share": -0.2, "effective_dim": 3.5, "following_mean": 0.32,
                    "following_by_author": {"planner": 0.32},
                    "liveness": {"E1": 1.0, "S1": 0.4}, "pool_spread": {"zpool": 0.03}},
        "vestibular": {"improvement_over_persistence": 0.00000007, "trunk_roll_deg": -12.0},
        "machine": {"commit_gb": 40.0, "commit_limit_gb": 68.8, "gpu_used_mib": 4000, "gpu_total_mib": 8188,
                    "step_ms": 800},
        "liveness": {"layers": {"tenure_heads.pose.0": 0.0, "forward_model.blocks.0.ffn.0": 0.99},
                     "pools": {"encoder_registry.encoders.motor.pool_attn": 0.001}, "line": -3, "samples": 24,
                     "revived": ["tenure_heads.pose.0"]},
        "self_touch_controlled": {"improvement_over_shuffled": -0.0117, "rows": 23},
        "self_contact_pairs": 0.4,
        "body_orientation": [{"valid": True, "roll_degrees": 3.0}, {"valid": True, "roll_degrees": -175.0}],
        "cmd_response_cos_held": 0.153,
    }


ARGS_ON = {"causal_chain": "on", "controllability": "on", "learning_progress": "on", "wellness_rank": "normalized",
           "contact_head": "on", "motor_synergy": "on", "vestibular_sense": "on", "efference_heads": "on"}


def probes_index(stand_in=True):
    heads = [
        (7, "gaze", {"metric": "gaze_captured_share_vision_150ms", "value": 7.314362812039169e-08,
                     "unit": "share of the real change in the vision token (0-1)", "floor": 0.0, "goal": 1.0,
                     "distance_to_goal": 0.9999999268563718,
                     "lived": "moved 7.3e-06% of the way; at 50 ms 3e-06%."}),
        (8, "body", {"metric": "body_captured_share_tenure_cm", "value": 0.0, "unit": "share (0-1)",
                     "floor": 0.0, "goal": 1.0, "distance_to_goal": 1.0, "lived": "foresaw 0.0% of it"}),
        (9, "binding", {"metric": "binding_vision_share_hand_entering_view", "value": 0.0013529091357736979,
                        "unit": "share of the real change in the vision token (0-1)", "floor": 0.0, "goal": 1.0,
                        "distance_to_goal": 0.9986470908642263, "lived": "foresaw 0.1%; error ratio 0.97"}),
    ]
    if stand_in:
        heads.append((10, "posture", {"metric": "posture_neck_share_50ms", "value": 2.2465571789339237e-08,
                                      "unit": "share of the real change in the proprioception token (0-1)",
                                      "floor": 0.0, "goal": 1.0, "distance_to_goal": 0.9999999775344283,
                                      "lived": "proxy: the head's row cannot be separated inside it"}))
    items = {}
    for item, probe, head in heads:
        latest = dict(head, lineage_step=400, checkpoint="x/ckpt_step_400", time="t", path="p.json")
        items[str(item)] = [dict(probe=probe, metric=head["metric"], unit=head["unit"], points=[[400, head["value"]]],
                                 latest=latest, words=ls.probe_words(latest, probe))]
    return {"items": items, "index_found": True}


# --------------------------------------------------------------------------- the Now cards
@pytest.mark.parametrize("record,args", [
    ({}, {}),
    ({"type": "mimo", "step": 1, "timestamp": 1, "prediction_audit": audit()}, {"causal_chain": "off"}),
    (None, ARGS_ON),
])
def test_every_now_card_prints_no_bare_decimal_raw_float_or_doubled_sentence(record, args):
    record = rich_record() if record is None else record
    panel = ls.gain_of_function(record, args, probes_index())
    assert panel["total"] == 10
    for row in panel["items"]:
        assert_plain(ls.item_text(row), f"item {row['item']}")
    head = ls.headline(panel, record, {"proprioception": dict(now=0.00016, n_now=161, lo=-0.0017, hi=0.0018,
                                                                n_pre=3000)})
    assert_plain(head["title"] + " " + head["detail"], "headline")


def test_each_measured_item_names_its_control_in_its_own_text():
    panel = ls.gain_of_function(rich_record(), ARGS_ON, probes_index())
    text = {r["item"]: ls.item_text(r) for r in panel["items"]}
    assert all(r["state"] == "measured" for r in panel["items"]), [r["item"] for r in panel["items"] if r["state"] != "measured"]
    assert "Picking at random among the 5 candidates" in text[1]
    assert "another body's command (the control)" in text[2]
    assert "No matched comparison" in text[3] and "learning disabled" in text[3]
    assert "blind to age" in text[4] and "of the ranking's weights" in text[4]
    assert "physics alone" in text[5]
    assert "base-rate guess" in text[6]
    for item in (7, 8, 9):
        assert "It does not test whether the model can choose and repeat a useful movement" in text[item]


def test_real_producer_summaries_reach_the_page_with_their_controls():
    """B1 (d): each producer's own summary() object through gain_of_function."""
    # The gate needs 32 independent orders, not 32 body-orders: the shuffled control
    # couples the bodies within each order. Allow for the unfinished first/last order.
    follows = dict(rich_record(), controllability=dict(controllability_summary(orders=40), difference_mean=0.99))
    row = {r["item"]: r for r in ls.gain_of_function(follows, ARGS_ON, {})["items"]}[2]
    assert row["state"] == "measured" and row["answer"].startswith("Better than a prediction from the same body state"), row["answer"]
    assert row["beats"] is True
    assert "100% of the way from 'unrelated to what it expected' to 'exactly as expected'" in ls.item_text(row)
    few = {r["item"]: r for r in ls.gain_of_function(rich_record(), ARGS_ON, {})["items"]}[2]
    assert few["beats"] is False and "has not been established by the recent statistical check" in few["answer"]
    still = dict(follows, controllability=controllability_summary(follow=0.0))
    row = {r["item"]: r for r in ls.gain_of_function(still, ARGS_ON, {})["items"]}[2]
    assert row["answer"].startswith("No advantage over a prediction from the same body state")
    # learning progress with no same-moment control: the frozen-weight band is printed beside it
    flat = dict(follows, learning_progress=dict(lp_summary(falling=False), value_mean=-0.03))
    row = {r["item"]: r for r in ls.gain_of_function(flat, ARGS_ON, {})["items"]}[3]
    assert "inside that band, so it is not evidence of learning" in ls.item_text(row)


def test_a_block_present_with_nothing_scored_says_so_and_why():
    """M3: the block is there and says why nothing was scored; it is not 'missing'."""
    rec = {"type": "mimo", "step": 1, "timestamp": 1,
           "controllability": dict(controllability_summary(orders=0), orders=78, efference_missing=78,
                                   measured_share=0.0)}
    row = {r["item"]: r for r in ls.gain_of_function(rec, {"controllability": "on"}, {})["items"]}[2]
    assert row["state"] == "unmeasured"
    assert "block is present but carries no controllability.agreement_mean" in row["unmeasured_because"]
    assert "78 commands graded" in row["unmeasured_because"]
    assert "without the body head's command-long expectation" in row["unmeasured_because"]
    # the producer's own reason (gate_why) and why an order is not scored (verify-fix-dashboards)
    assert "0 of the 32 orders the gate judges" in row["unmeasured_because"]      # the trainer's own gate_why text
    assert "beside the shuffled-order control" in row["unmeasured_because"]       # the same
    assert "rise above the body's drift while holding still" in row["unmeasured_because"]
    assert_plain(row["unmeasured_because"], "controllability, nothing scored")
    numeric_why = dict(rec["controllability"], gate_why="over the last 32 measured orders the result agreed "
                       "0.0010 more than with another body's, under the line of 0.0200")
    row = {r["item"]: r for r in ls.gain_of_function(dict(rec, controllability=numeric_why),
                                                     {"controllability": "on"}, {})["items"]}[2]
    assert "0.0010" not in row["unmeasured_because"] and "its evidence gate shut" in row["unmeasured_because"]
    rec2 = {"type": "mimo", "step": 1, "timestamp": 1}
    row = {r["item"]: r for r in ls.gain_of_function(rec2, {"controllability": "on"}, {})["items"]}[2]
    assert "is on but the newest record carries no 'controllability' block" in row["unmeasured_because"]
    row = {r["item"]: r for r in ls.gain_of_function(rec2, {"wellness_rank": "raw"}, {})["items"]}[4]
    assert "--wellness-rank is off (set to raw)" in row["unmeasured_because"]


def test_item_1_has_no_goal_line_counts_ties_and_shows_the_gap():
    """M1 and reviews/b1 MAJOR 3 and MINOR 2."""
    row = {r["item"]: r for r in ls.gain_of_function(rich_record(), ARGS_ON, {})["items"]}[1]
    text = ls.item_text(row)
    assert "No validated success target: the planner does not aim at predicted wellness" in text
    # what a tie is (own-command analysis review, 2026-10-02): the band 0.016 against the spread 0.005 of this record
    assert ("Tied in 55 of every 100 plans (a tie: the chosen command's predicted wellness sum lies within the tie "
            "band of the best alternative's; the band is the evaluator's own error on a difference between two "
            "candidates, 0.32% of the 0-to-5 wellness scale, 3.2 times the spread predicted between the candidates in "
            "this window, so every choice reads tied whatever the forward model predicts; the tied share is no evidence "
            "about the forward model: its planner-side reading is the planner recheck's captured share on the "
            "Development page)") in text
    assert "a tie is not counted as choosing the best" in text
    assert "of the full score range (0 to 5)" in text
    assert "1.4 cm from the command the planner imagined" in text
    assert "planner_share_of_executed" not in json.dumps(ls.SERIES["cc.plan_exec_cm"], default=str)
    assert "cc.planner_executed" not in ls.SERIES
    older = dict(rich_record(), causal_chain=chain_summary(chosen_best_share=1.0, alternatives_n=4, planner_events=3))
    text = ls.item_text({r["item"]: r for r in ls.gain_of_function(older, ARGS_ON, {})["items"]}[1])
    assert "Ties are not reported by this record" in text and "short of" not in text and "past the goal" not in text


def test_item_4_sets_lead_share_against_weight_share_and_reads_goal_age():
    """M4 and reviews/b3: shares that add to 100 can exceed it; goal age with its age-blind control."""
    row = {r["item"]: r for r in ls.gain_of_function(rich_record(), ARGS_ON, {})["items"]}[4]
    text = ls.item_text(row)
    assert re.search(r"integrity: \d+ of every 100 parts of the lead, while it holds 44 of every 100 parts", text)
    assert "can be below 0 or above 100" in text
    assert "93% of the way back through the list of remembered body states" in text and "95 of every 100 goals" in text
    assert row["answer"].startswith("Choices concentrate on older or newer experiences.")
    big = dict(rich_record(), wellness_rank=dict(rank_block(), winner_margin_share={"integrity": 1.28, "coherence": -0.28},
                                                 weight_share_by_channel={"integrity": 0.28, "coherence": 0.07}))
    text = ls.item_text({r["item"]: r for r in ls.gain_of_function(big, ARGS_ON, {})["items"]}[4])
    assert "128 of every 100 parts of the lead, while it holds 28 of every 100 parts" in text
    assert "lower is better" not in text and "decides alone" not in text


def test_item_5_reads_following_never_the_reconstruction():
    """M5: the map's reconstruction is logged beside the item, not its measurement."""
    row = {r["item"]: r for r in ls.gain_of_function(rich_record(), ARGS_ON, {})["items"]}[5]
    assert ls.ITEMS[4]["key"] == "syn.following"
    assert row["big"] == "32%"
    lines = {ln["kind"]: ln["text"] for ln in row["lines"]}
    assert "reproduces -20% of the way" in lines["related"]
    by_author = dict(rich_record(), synergy=dict(rich_record()["synergy"], following_by_author={"planner": 0.6}))
    row = {r["item"]: r for r in ls.gain_of_function(by_author, ARGS_ON, {})["items"]}[5]
    assert row["big"] == "60%" and "the planner's synergy-written commands" in ls.item_text(row)


def test_item_6_reads_onsets_against_the_base_rate_then_falls_back_to_paired_onsets():
    """B1, reviews/b4 M3 and M4: onsets only; success_paired against success_control, never
    success_self; habituation 1 = a new pair."""
    text = ls.item_text({r["item"]: r for r in ls.gain_of_function(rich_record(), ARGS_ON, {})["items"]}[6])
    assert "74 of every 100 pairs" in text and "scored 82 of every 100 pairs" in text
    assert "No advantage over a base-rate guess" in text
    assert "100% = a new pair" in text and "Own body: no; surroundings: yes" in text
    fallback = dict(rich_record(), contact_control=contact_summary(success_paired=0.53, success_control=0.47,
                                                                    success_self=0.21, control_onsets_self=30))
    row = {r["item"]: r for r in ls.gain_of_function(fallback, ARGS_ON, {})["items"]}[6]
    text = ls.item_text(row)
    assert "53 of every 100 new self-touches with its own command" in text
    assert "another body's command foresaw 47 of every 100" in text and "21 of every 100" not in text
    assert "No success threshold has been validated" in text


def test_item_7_shows_the_eye_angles_beside_the_vision_probe_and_the_probe_as_a_percentage():
    """reviews/b1 MAJOR 4 and reviews/dashboards.md M7."""
    row = {r["item"]: r for r in ls.gain_of_function(rich_record(), ARGS_ON, probes_index())["items"]}[7]
    text = ls.item_text(row)
    assert row["big"] == "0.0000073% of the way"
    assert "Beside it, eye angles alone" in text and "3 degrees against 6 degrees" in text
    assert "0.9999999268563718" not in text and "7.314362812039169" not in text
    assert "token" not in row["big"] and "handle" not in text.split("The probe's own summary")[0]
    assert "The probe's own summary: moved 0.0000073% of the way" in text


def test_item_10_with_the_balance_sense_off_is_unmeasured_and_its_stand_in_is_not_counted():
    """M2: a stand-in probe reading never takes the balance sense's name or count, and the flag
    that measures item 10 stays in 'What happens next'."""
    panel = ls.gain_of_function({"type": "mimo", "step": 1, "timestamp": 1}, {}, probes_index())
    row = {r["item"]: r for r in panel["items"]}[10]
    assert row["state"] == "stand-in" and not row["measured"]
    assert "disabled or was not configured" in row["answer"] and "stand-in" in row["answer"]
    assert panel["measured"] == 3 and panel["stand_in"] == 1
    head = ls.headline(panel, {})
    assert head["title"].startswith("No gain-of-function item is shown working yet")
    assert "1 item has only a stand-in reading, not counted" in head["detail"]
    assert "0.14% of the way" in head["detail"]


def test_the_headline_section_20_sentence_depends_on_the_value_and_a_measured_line():
    """M6: inside the pre-build windows' spread the section-20 sentence is printed; above it, not."""
    panel = ls.gain_of_function({}, {}, {})
    inside = ls.headline(panel, {}, {"tactile": dict(now=0.0001, n_now=100, lo=-0.01, hi=0.01, n_pre=3000)})
    assert "section 20 treats it as evidence" in inside["detail"]
    above = ls.headline(panel, {}, {"tactile": dict(now=0.05, n_now=100, lo=-0.01, hi=0.01, n_pre=3000)})
    assert "section 20" not in above["detail"] and "above that range" in above["detail"]
    alone = ls.headline(panel, {"prediction_audit": audit()})
    assert "one window is too noisy to judge" in alone["detail"]


def test_the_probe_index_keeps_the_newest_line_per_probe_and_checkpoint(tmp_path):
    """reviews/probes.md and m5: the re-run of the gaze probe replaces the first run."""
    rows = [{"probe": "gaze", "checkpoint": "c1", "time": "2026-09-22 19:28:29", "headline": {"value": 1}},
            {"probe": "body", "checkpoint": "c1", "time": "2026-09-22 19:34:30", "headline": {"value": 2}},
            {"probe": "gaze", "checkpoint": "c1", "time": "2026-09-22 19:43:41", "headline": {"value": 3}},
            {"probe": "gaze", "checkpoint": "c2", "time": "2026-09-22 20:00:00", "headline": {"value": 4}}]
    kept = ls.latest_index(rows)
    assert [(r["probe"], r["checkpoint"], r["headline"]["value"]) for r in kept] == [
        ("body", "c1", 2), ("gaze", "c1", 3), ("gaze", "c2", 4)]


def test_sensitivity_is_not_tied_to_the_causal_chain_flag():
    """reviews/b1 MINOR 10."""
    for sense, _ in ls.SENSES:
        assert ls.SERIES[f"sens.{sense}"]["flag"] is None


def test_launch_settings_compare_values_in_force_and_label_env_settings(monkeypatch):
    """M8: a flag absent from the parent ran at its default; an env setting changed at a resume
    before the migration fix was ignored by the resumed worlds."""
    monkeypatch.setattr(ls, "commit_has", lambda commit, fix=None, root=None: False)
    defaults = {"finger_armature": 0.0004, "toy_hand_off": "off", "babble_noise": 0.05}
    diff = ls.args_difference({"finger_armature": 0.0004, "toy_hand_off": "follow"}, {}, defaults, commit="30c0c7a8e")
    assert [d["flag"] for d in diff] == ["--toy-hand-off"]
    assert "ignored by the resumed worlds" in diff[0]["note"]
    assert diff[0]["text"].startswith("--toy-hand-off follow (was --toy-hand-off off, the trainer's default")
    # a commit between bdbfb7895 (the resume applies the launch's toy hand-off) and 04c84682f
    monkeypatch.setattr(ls, "commit_has", lambda commit, fix=ls.ENV_FIX_COMMIT, root=None: fix == ls.ENV_FIX_COMMIT)
    diff = ls.args_difference({"toy_hand_off": "follow"}, {"toy_hand_off": "off"}, defaults, commit="later")
    assert "applied to the resumed worlds (commit bdbfb7895 migrates it)" in diff[0]["note"]
    assert "uncommitted changes not recorded" in ls.code_text("30c0c7a8e")
    assert ls.trainer_defaults().get("toy_hand_off") is not None, "defaults are read from the trainer's source"


def test_every_setting_a_resume_can_migrate_is_labelled_as_applied(monkeypatch):
    """The page's list of migrated env settings is the resume's own list: the finger repair's
    hand_target_mode and the motor interface fixes' three flags were deployed with
    --resume-env-settings launch, and the page must not call them ignored by the resumed worlds."""
    from mimo_bridge import continuation_state as CS
    assert set(ls.ENV_MIGRATED) == set(CS.ENV_KWARGS_MIGRATABLE)
    monkeypatch.setattr(ls, "commit_has", lambda commit, fix=None, root=None: True)
    defaults = {"position_rows": "all", "hand_target_mode": "world"}
    diff = ls.args_difference({"position_rows": "no_head_pelvis", "hand_target_mode": "wrist_relative_v1",
                               "resume_env_settings": "launch"},
                              {"resume_env_settings": "launch"}, defaults, commit="later")
    notes = {d["flag"]: d.get("note", "") for d in diff}
    for flag in ("--position-rows", "--hand-target-mode"):
        assert notes[flag] == "applied to the resumed worlds (--resume-env-settings launch)", notes
    kept = ls.args_difference({"position_rows": "no_head_pelvis"}, {}, defaults, commit="later")
    assert "not applied" in kept[0]["note"], "a plain resume keeps the saved worlds' setting"


# --------------------------------------------------------------------------- verify-fix-dashboards
# Written 2026-09-23 by verify-fix-dashboards: each test failed on 46844ce0c and passes on the repair.
def test_a_reading_inside_its_controls_noise_is_not_counted_as_beating_it():
    """B1 residual: agreement 0.011 against a control of 0.010 - the review's own "inside its
    control" numbers - with the producer's evidence gate shut read "Beats its control" and turned
    the headline into "Some items beat their controls". The same bare ">" judged items 3 and 6;
    item 6's pooled pay gate opens on touches by the world and is not evidence about self-touch."""
    rec = {"type": "mimo", "step": 1, "timestamp": 1,
           "controllability": {"agreement_mean": 0.011, "control_mean": 0.010, "measured_share": 0.4,
                               "orders": 78, "gate_open": False},
           "learning_progress": {"value_mean": 0.012, "control_mean": 0.006},
           "contact_control": {"onset_auc_self": 0.83, "control_onset_auc_self": 0.82, "gate_open": True}}
    panel = ls.gain_of_function(rec, {"controllability": "on", "learning_progress": "on", "contact_head": "on"}, {})
    by = {r["item"]: r for r in panel["items"]}
    for item in (2, 3, 6):
        assert by[item]["state"] == "measured" and by[item]["beats"] is False, (item, by[item]["answer"])
        assert "has not been established" in by[item]["answer"] or "no statistical check" in by[item]["answer"], (item, by[item]["answer"])
        assert_plain(ls.item_text(by[item]), f"item {item}")
    head = ls.headline(panel, rec)
    assert head["title"].startswith("No gain-of-function item is shown working yet"), head["title"]
    assert "0 of the 3 items measured against a control beat it" in head["detail"]
    opened = dict(rec, contact_control=dict(rec["contact_control"], gate_self_open=True))
    row = {r["item"]: r for r in ls.gain_of_function(opened, {"contact_head": "on"}, {})["items"]}[6]
    assert row["beats"] is True and row["answer"].startswith("Better than a base-rate guess; the recent statistical check")


def test_the_headline_never_places_a_control_judged_item_against_knowing_nothing():
    """B1, M1 and M6 residual: the headline read "the highest reads 85% of the way from knowing
    nothing" for a contact head below its base-rate control (0.85 against 0.95; knowing nothing on
    the onset score is a coin flip, 50), and "100% of the way" for item 1 when a dead evaluator
    ties every candidate."""
    worse = {"type": "mimo", "step": 1, "timestamp": 1,
             "contact_control": {"onset_auc_self": 0.85, "control_onset_auc_self": 0.95}}
    head = ls.headline(ls.gain_of_function(worse, {"contact_head": "on"}, {}), worse)
    assert "85%" not in head["detail"] and "knowing nothing" not in head["detail"], head["detail"]
    assert "1 of 10 items has a direct measurement" in head["detail"]
    tied = {"type": "mimo", "step": 1, "timestamp": 1,
            "causal_chain": {"chosen_best_share": 1.0, "alternatives_n": 4, "planner_events": 20}}
    head = ls.headline(ls.gain_of_function(tied, {"causal_chain": "on"}, {}), tied)
    assert "100%" not in head["detail"] and head["title"].startswith("No gain-of-function item is shown working")
    assert_plain(head["title"] + " " + head["detail"], "headline")
    head = ls.headline(ls.gain_of_function({}, {}, probes_index()), {})
    assert ("The largest probe reading is 0.14% of the way from 'the prediction ignores the command' to 'it "
            "foresees the whole change'") in head["detail"]


def test_a_plain_resume_on_code_after_04c84682f_applies_no_env_setting(monkeypatch):
    """M8 residual: since 04c84682f (fix-ops, DESIGN D2 note) a resume keeps the saved worlds'
    settings unless it is launched with --resume-env-settings launch; the page said "applied to
    the resumed worlds" for any commit containing bdbfb7895."""
    monkeypatch.setattr(ls, "commit_has", lambda commit, fix=ls.ENV_FIX_COMMIT, root=None: True)
    defaults = {"toy_hand_off": "off"}
    diff = ls.args_difference({"toy_hand_off": "follow"}, {"toy_hand_off": "off"}, defaults, commit="head")
    assert "not applied" in diff[0]["note"] and "--resume-env-settings launch" in diff[0]["note"]
    assert "applied to the resumed worlds" not in diff[0]["note"]
    diff = ls.args_difference({"toy_hand_off": "follow", "resume_env_settings": "launch"}, {"toy_hand_off": "off"},
                              defaults, commit="head")
    toy = next(d for d in diff if d["flag"] == "--toy-hand-off")
    assert toy["note"] == "applied to the resumed worlds (--resume-env-settings launch)"
    monkeypatch.setattr(ls, "commit_has", lambda commit, fix=ls.ENV_FIX_COMMIT, root=None: None)
    diff = ls.args_difference({"toy_hand_off": "follow"}, {"toy_hand_off": "off"}, defaults, commit=None)
    assert "unknown" in diff[0]["note"]


def test_each_new_module_has_its_own_liveness_series_on_over_time():
    """M10 residual ("Add a lineage series per module"): the new modules' liveness and pool spread
    are series of their own, charted on the liveness card, not only folded into one count."""
    rec = rich_record()
    assert ls.SERIES["live.contact_control"]["get"](rec) == 0.2
    assert ls.SERIES["live.synergy"]["get"](rec) == 0.4
    assert ls.SERIES["pool.synergy"]["get"](rec) == 0.03
    assert ls.SERIES["live.efference"]["get"](rec) is None, "absent is unmeasured, never 0"
    assert {"live.efference", "live.contact_control", "live.synergy"} <= set(ls.chart_keys())
    assert {"pool.synergy", "live.vestibular"} <= set(ls.other_keys())


def test_a_reconstruction_level_outside_the_undamaged_range_is_never_green():
    """The same class as M9 (a green chip on a measured failure): segment 31 read body-position
    reconstruction at 47 times next-state error (vision 16 times) under residual prediction, where
    the undamaged residual runs measured 1.16 to 2.45 times, and the chip read green "steady"; the
    lead's sweep found those decoders 17 and 10 in 100 units responding (DESIGN.md section 10)."""
    from mimo_bridge.tools import dashboard
    seg31 = {m: dict(recon=r * n, next=n, ratio=r, baseline=r * n, residual=True, inverted=False)
             for m, r, n in (("vision", 15.64, 0.00938), ("tactile", 0.41, 0.02325), ("proprioception", 46.6, 0.00387))}
    chips = dashboard.chips_for({"recon_health": seg31})
    assert "steady" not in chips and 'class="chip g' not in chips, chips
    assert "47 times next-state error on proprioception" in chips and "1.2 to 2.5 times" in chips
    assert not bare_decimals(page_text(chips)) and not banned_words(page_text(chips))
    undamaged = {m: dict(v, ratio=2.0) for m, v in seg31.items()}
    assert "reconstruction steady" in dashboard.chips_for({"recon_health": undamaged})


def test_a_negative_probe_reading_says_what_below_zero_means():
    """The gaze probe re-run of 2026-09-23 01:46 reads -0.0000007% on ckpt_step_344400; the card
    printed the minus sign without saying the prediction moved the wrong way."""
    head = {"metric": "gaze_captured_share_vision_150ms", "value": -6.9720859563020695e-09,
            "unit": "share of the real change in the vision token (0-1)", "floor": 0.0, "goal": None}
    w = ls.probe_words(head, "gaze")
    assert w["big"] == "-0.0000007% of the way"
    assert "below 0% = the prediction moved the opposite way from the real change" in w["scale"]
    assert "below 0%" not in ls.probe_words(dict(head, value=7.3e-8), "gaze")["scale"]


def test_the_old_dashboard_gates_table_uses_no_status_shorthand():
    """the style guide: "armed" is status shorthand; the planner gates say what turns them on."""
    from mimo_bridge.tools import dashboard
    switches, _settings = dashboard.gates_from({"intention": "on"}, {}, "no data")
    states = {n: s for n, s, _d in switches}
    assert states["plan depth cap"] == "on (the planner is on)"
    assert not banned_words(" ".join(states.values()))
    chips = dashboard.chips_for({"gates": (switches, [])})
    assert "gates: 4 on" in chips, chips          # intention and the three planner gates


def test_item_5_every_order_following_is_a_stand_in_until_the_record_splits_it_by_author():
    """M5 residual (verify-fix-dashboards, resumed 2026-09-23): b5's synergy block writes
    following_mean over EVERY graded order, babble and reflex included (lem_model.py
    _synergy_grade adds each order's direction match whoever wrote it). The item counted it as
    the synergy's measurement, so an every-order reading past the physics-only reference read
    "At or past the goal line" and the headline said item 5 reached its measured goal line. The
    field contract names the item's measurement: following by author (or b5's physics check)."""
    rec = {"type": "mimo", "step": 1, "timestamp": 1, "exec_source": {"planner": 0.1, "reflex_noise": 0.9},
           "synergy": {"dim": 32, "following_mean": 0.95, "explained_share": 0.4}}
    panel = ls.gain_of_function(rec, {"motor_synergy": "on"}, {})
    row = {r["item"]: r for r in panel["items"]}[5]
    assert row["state"] == "stand-in" and not row["measured"], row["answer"]
    assert row["big"] == "Indirect test only"
    text = ls.item_text(row)
    assert "following_by_author" in row["answer"] and "not counted" in row["answer"]
    assert "whoever wrote it" in text and "10 of every 100 latched commands" in text
    assert_plain(text, "item 5 stand-in")
    head = ls.headline(panel, rec)
    assert "goal line in the newest window" not in head["title"], head["title"]
    assert "1 item has only a stand-in reading, not counted" in head["detail"]
    assert ls.SERIES["syn.following"]["get"](rec) is None, "the item's series reads only the split by author"
    assert ls.SERIES["syn.following_all"]["get"](rec) == 0.95
    split = dict(rec, synergy=dict(rec["synergy"], following_by_author={"planner": 0.95}))
    row = {r["item"]: r for r in ls.gain_of_function(split, {"motor_synergy": "on"}, {})["items"]}[5]
    assert row["state"] == "measured" and row["answer"] == "At or past the goal line."
    from mimo_bridge.tools import dashboard
    old = dict(dashboard.gof_rows(rec))["Motor synergy map"]
    assert "whoever wrote it" in old and not bare_decimals(old)


def test_learning_progress_above_the_frozen_band_by_less_than_its_width_is_not_told_from_noise():
    """B1 (b) residual: frozen weights read -7.1% to -1.6% (b3 measurement 2), a spread of 5.5
    percentage points from the world changing alone; the review names a training reading of 0.6%
    to 1.2% as inside that spread. It printed "Outside the frozen-weight band"."""
    def row_for(v):
        rec = {"type": "mimo", "step": 1, "timestamp": 1, "learning_progress": {"value_mean": v}}
        return {r["item"]: r for r in ls.gain_of_function(rec, {"learning_progress": "on"}, {})["items"]}[3]
    # The old historical range is unmatched: neither above nor below it establishes learning.
    for value in (.006, .1, -.2, -.03):
        row = row_for(value)
        assert row['answer'].startswith('Learning is not established by this score.')
        assert row['beats'] is False
        assert 'not a matched comparison' in ls.item_text(row)
        assert_plain(ls.item_text(row), 'item 3')


def test_the_drift_chip_says_times_its_first_value_never_a_bare_ratio():
    """the style guide: raw ratios without anchors are banned from every dashboard row. The old chip read
    "ungraded-channel drift x1.0" (x0.8 below 1 escapes the decimal scanner: a letter precedes it)."""
    from mimo_bridge.tools import dashboard
    for ratio in (0.8, 1.0, 4.2):
        chips = dashboard.chips_for({"drift": {"intermind": {"now": ratio, "start": 1.0, "ratio": ratio}}})
        text = page_text(chips)
        assert not re.search(r"\bx\d", text), text
        assert ("the same as its first value" if ratio == 1.0 else "times its first value") in text, text
        assert "over 3 times" in text, text
        assert not bare_decimals(text)


def _probe_index(tmp_path, rows):
    run = tmp_path / "Saved/mimo_bridge/experiments/R"
    run.mkdir(parents=True, exist_ok=True)
    index = tmp_path / "Saved/mimo_bridge/gof_probes/index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("\n".join(json.dumps(dict(r, run="R")) for r in rows) + "\n", encoding="utf-8")
    return ls.probe_results(tmp_path, {str(run): None})


def test_a_probe_whose_newest_run_failed_is_not_counted_as_measured(tmp_path):
    """reviews/probes.md blocker class: the suite writes a failure record (value None, status
    "failed: ...") into index.jsonl when a probe stops with an error, and the newest line per
    (probe, checkpoint) replaced the measurement. The item then counted as measured, printed
    "unmeasured" as its value and said "the reading is on its probe's own scale"
    (verify-fix-dashboards, resumed 2026-09-23)."""
    fail = "failed: RuntimeError: Unserialized copied model state: contact_tracker (ContactTracker)"
    probes = _probe_index(tmp_path, [
        {"probe": "gaze", "item": 7, "checkpoint": "x/ckpt_step_100", "lineage_step": 100, "time": "2026-09-23 01:00:00",
         "status": "measured", "headline": {"metric": "gaze_captured_share_vision_150ms", "value": 1e-7,
                                            "unit": "share", "floor": 0.0, "goal": None}},
        {"probe": "gaze", "item": 7, "checkpoint": "x/ckpt_step_200", "lineage_step": 200, "time": "2026-09-23 02:00:00",
         "status": fail, "headline": {"metric": "gaze_unmeasured", "value": None, "unit": "",
                                      "lived": "unmeasured: the probe stopped with an error (see status)"}},
        {"probe": "body", "item": 8, "checkpoint": "x/ckpt_step_200", "lineage_step": 200, "time": "2026-09-23 02:01:00",
         "status": fail, "headline": {"metric": "body_unmeasured", "value": None, "unit": "",
                                      "lived": "unmeasured: the probe stopped with an error (see status)"}},
        {"probe": "binding", "item": 9, "checkpoint": "x/ckpt_step_200", "lineage_step": 200, "time": "2026-09-23 02:56:37",
         "status": "measured", "headline": {
             "metric": "binding_vision_counterfactual_hand_entering_view", "value": -0.004719146254189657,
             "unit": "share of the difference in what the eyes saw between continuing the command and stopping the hand",
             "floor": 0.0, "goal": None, "lived": "moved -0.472% (5 in every thousand) of the way"}}])
    panel = ls.gain_of_function({}, {}, probes)
    by = {r["item"]: r for r in panel["items"]}
    assert by[8]["state"] == "unmeasured" and not by[8]["measured"], by[8]["answer"]
    assert "stopped with an error" in by[8]["answer"] and "contact_tracker" in ls.item_text(by[8])
    assert by[7]["state"] == "measured" and by[7]["value"] == 1e-7
    text7 = ls.item_text(by[7])
    assert "(lineage step 100)" in text7 and "The newest run of this probe" in text7 and "stopped with an error" in text7
    assert panel["measured"] == 2, [(r["item"], r["state"]) for r in panel["items"]]
    text9 = ls.item_text(by[9])
    assert "continuing the command" in text9 and "stopping the hand" in text9 and "binding:" not in text9, text9
    assert "below 0% = the prediction moved the opposite way" in text9
    for row in panel["items"]:
        assert_plain(ls.item_text(row), f"item {row['item']}")
    assert [e.get("points") for e in probes["items"]["8"]] == [[]]
    head = ls.headline(panel, {})
    assert "2 of 10 items have a direct measurement" in head["detail"], head["detail"]


# --------------------------------------------------------------------------- Over time
def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _lineage(root, child_args=None, records=None):
    saved = root / "Saved/mimo_bridge"
    parent, child = saved / "experiments/H1", saved / "experiments/H1_seg2"
    _write(parent / "args.json", {"n_envs": 2})
    _write(child / "args.json", dict({"n_envs": 2, "resume": str(parent / "checkpoints/ckpt_step_300")},
                                     **(child_args or {})))
    for run, steps in ((parent, (100, 200, 300)), (child, (400, 500))):
        extra = {k: v for k, v in (records or {}).items() if k not in ("type", "step", "timestamp")}
        rows = [dict({"type": "mimo", "step": s, "timestamp": 1_790_000_000 + s, "model_steps_per_s": 0.4,
                      "prediction_audit": audit(), "cmd_response_cos_held": 0.15}, **extra) for s in steps]
        (run / "replay_training_stats.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                                          encoding="utf-8")
    _write(saved / "training_continuation.json", {"active_run": str(child), "timestamp": 1_790_000_600,
                                                  "status": "training"})
    return saved


def test_every_over_time_panel_sentence_passes_and_says_why_a_series_is_absent(tmp_path):
    """M3 on Over time: the reason reads each segment's launch settings."""
    _lineage(tmp_path, child_args={"causal_chain": "on"})
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys="all", now=1)
    for card in out["cards"]:
        for panel in card["panels"]:
            assert_plain(" ".join(v for v in panel["texts"].values()), f"{card['id']}/{panel['title']}")
    for panel in out["more"]:
        assert_plain(" ".join(v for v in panel["texts"].values()), f"more/{panel['title']}")
    texts = {p["title"]: p["texts"] for c in out["cards"] for p in c["panels"]}
    assert "segment seg2 ran with --causal-chain on, but no record carries" in \
        texts["It chose the plan it predicted would feel best"]["unmeasured"]
    assert "no segment's launch settings (args.json) turn it" in texts["What the body did matches what it expected"]["unmeasured"]
    assert "15% of the way from 'the turn is unrelated to the command'" in \
        texts["The body turns the way it was commanded (every command)"]["latest"]
    json.dumps(out, allow_nan=False)


def test_an_over_time_card_says_one_unmeasured_reason_once(tmp_path):
    """verify-fix-dashboards (reading gate): on the real lineage, where no segment ran a new flag,
    the item-5 card printed "Unmeasured on every segment of this lineage: it is written only while
    --motor-synergy is on, ..." three times (one per panel), items 1, 2, 4 and 6 and the liveness
    card twice. Each card is scanned whole, as the page shows it."""
    _lineage(tmp_path)
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", now=1)
    for card in out["cards"]:
        text = "\n".join([card["title"], card["what"]] + [p["title"] + "\n" + "\n".join(p["texts"].values())
                                                          for p in card["panels"]])
        assert not doubled_sentences(text), (card["id"], doubled_sentences(text)[:2])
    syn = next(c for c in out["cards"] if c["id"] == "item5")
    reasons = [p["texts"].get("unmeasured") for p in syn["panels"]]
    assert reasons[0].startswith("Unmeasured on every segment") and reasons[1:] == ["Unmeasured, as above."] * (
        len(reasons) - 1) and len(reasons) >= 3


def test_the_machine_card_charts_memory_against_its_limits(tmp_path):
    """m6: card memory against the card's total, commit charge against the machine's limit."""
    keys = ls.chart_keys()
    assert {"machine.gpu", "machine.gpu_total", "machine.commit", "machine.commit_limit"} <= set(keys)
    assert ls.SERIES["machine.gpu"]["limit_series"] == "machine.gpu_total"
    _lineage(tmp_path, records={"machine": {"commit_gb": 40.0, "commit_limit_gb": 68.8, "gpu_used_mib": 4096,
                                            "gpu_total_mib": 8192, "step_ms": 800}})
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", now=1)
    card = next(c for c in out["cards"] if c["id"] == "machine")
    latest = {p["title"]: p["texts"].get("latest", "") for p in card["panels"]}
    assert "4 GiB" in latest["Card memory in use, against the card's total"]
    assert "against the card's memory of 8 GiB" in latest["Card memory in use, against the card's total"]
    assert "against the machine's commit limit of 69 GiB" in latest["Commit charge, against the machine's limit"]
    assert ls.other_keys(), "every series no card shows is offered under More gauges"


# --------------------------------------------------------------------------- liveness, live tab, page layout
def test_the_liveness_card_lists_every_report_against_the_line_and_the_known_dead_parts():
    """M10."""
    card = dd.liveness_card(rich_record())
    text = " ".join([card["title"]] + [ln["text"] for ln in card["lines"]])
    assert card["title"].startswith("3 of 6 layer response readings have under half their units responding")
    assert "In the model's report, the tenure pose head: 0 of every 100 units respond" in text
    # the report is named by its module (lineage_series.LIVE_MODULES), never by its record key; the
    # fixture's layer keys are its own inventions, which the page prints as they are
    assert "In the contact head's report, trunk.0: 20 of every 100 units respond" in text
    assert "In the synergy map's report, zpool: attention spread 3% of an even spread" in text
    assert "No line is drawn for attention spread" in text
    assert "Revived by the repair: the tenure pose head" in text
    assert_plain(text, "liveness card")
    known = dd.liveness_card({})
    text = " ".join(ln["text"] for ln in known["lines"])
    assert "the lead's sweep of ckpt_step_344400" in text
    assert "The tenure pose head: 0 of every 100 units respond" in text
    assert "locked on one, DESIGN.md section 10" in text
    assert "Unread: eyes (the gaze command)" in text
    assert_plain(text, "known liveness")


def test_the_live_tab_says_not_publishing_for_an_old_picture_and_prints_plain_sentences(tmp_path):
    """m2: a picture older than three live-view periods is not publishing."""
    run = tmp_path / "Saved/mimo_bridge/experiments/R"
    (run / "live").mkdir(parents=True)
    (run / "live" / "frame.jpg").write_bytes(b"x")
    old = 1_790_000_000
    os.utime(run / "live" / "frame.jpg", (old, old))
    training = {"run": "Saved/mimo_bridge/experiments/R", "run_name": "R"}
    row = dict(rich_record(), model_steps_per_s=1.0, self_contact_force_share=0.057)
    live = dd.live_state(tmp_path, training, row, {"live_view_every": 1})
    assert live["publishing"] is False
    assert live["texts"]["pub"].startswith("no: the last picture is") and "three live-view periods, 3 seconds" in live["texts"]["pub"]
    assert_plain(" ".join(live["texts"].values()), "live tab")
    os.utime(run / "live" / "frame.jpg", None)
    assert dd.live_state(tmp_path, training, row, {"live_view_every": 1})["publishing"] is True


def test_the_page_moves_the_2026_09_13_registries_into_earlier_lineages_and_prints_server_sentences():
    """m7, and the Now cards lay out the server's lines (no string building of numbers)."""
    root = Path(dd.__file__).parent
    page = (root / "development_dashboard.html").read_text(encoding="utf-8")
    now = page.split('<section id="history"')[0]
    earlier = page.split('<details id="earlier-lineages"')[1].split("</details>")[0]
    assert 'id="development-stages"' not in now and 'id="ability-rows"' not in now
    assert 'id="development-stages"' in earlier and 'id="ability-rows"' in earlier
    assert 'id="liveness-lines"' in now and 'id="more-gauges"' in page
    js = (root / "development_dashboard.js").read_text(encoding="utf-8")
    gof = js.split("function gofItem(row){")[1].split("\n}\n")[0]
    assert "toFixed" not in gof and "row.lines" in gof
    assert "toFixed(4)" not in js and ".toFixed(3)" not in js, "raw score formatting is gone from the page script"


def test_the_now_state_prints_plain_sentences_on_a_fixture(tmp_path):
    saved = _lineage(tmp_path, records=rich_record())
    reg = tmp_path / "docs/development_dashboard/evidence.json"
    _write(reg, {"schema": 1, "abilities": [], "records": []})
    state = dd.build_state(tmp_path, now=1_790_000_700)
    texts = [state["verdict"]["sentence"], *state["verdict"]["facts"], state["headline"]["title"],
             state["headline"]["detail"], state["what_changed"] or "", state["what_next"] or "",
             state["liveness"]["title"], *[ln["text"] for ln in state["liveness"]["lines"]],
             *[" ".join([c["title"], c["value"], *c["notes"]]) for c in state["observations"]],
             *state["live"]["texts"].values()]
    for t in texts:
        assert_plain(t, "Now")
    for row in state["gain_of_function"]["items"]:
        assert_plain(ls.item_text(row), f"item {row['item']}")
    assert "--vestibular-sense" in (state["what_next"] or ""), "no flag is on in this fixture's launch settings"


# --------------------------------------------------------------------------- the old dashboard and lifetime
def _popups(h):
    m = re.search(r"var D=(\{.*?\});function op", h, re.S)
    return "" if not m else "\n".join(v[f] for v in json.loads(m.group(1)).values() for f in ("t", "w", "h", "f"))


def test_a_commit_hash_of_digits_around_one_e_on_the_old_dashboards_chips_is_no_number():
    """The test below reads the checkout's real HEAD into the old dashboard's chips. At commit 2e3746163 (Oct 5, 2026,
    the registered unit run of 1:19 PM PDT) the chip read "HEAD 2e374616" and the scanner took it for a number in
    exponent form, so that test passed or failed with the commit's name. This one holds the chips' two sentences with
    such a hash, whatever HEAD is; a number in exponent form is still found."""
    for chip in ("code: run deadbeef != HEAD 2e374616", "the run's code does NOT match the current source HEAD 2e374616",
                 "code: run == HEAD (2e3746163)", "code: run 1234e567 (HEAD unknown)"):
        assert bare_decimals(page_text(chip)) == [], chip
    assert [n for n, _ in bare_decimals("code: run 7.3e-06 != HEAD 1e5")] == ["7.3e-06", "1e5"]
    # the page cleaner the Now page's records and the old dashboard's notes pass through printed such a hash as "inf"
    from mimo_bridge.tools.plain_numbers import clean, plain_tree
    assert clean("the fix at commit 2e3746163 holds") == "the fix at commit 2e3746163 holds"
    assert plain_tree({"commit": "2e3746163", "note": "HEAD 2e374616"}) == {"commit": "2e3746163",
                                                                            "note": "HEAD 2e374616"}
    assert clean("foresaw 7.3e-06% of the change") == "foresaw 0.0000073% of the change"


def test_the_old_dashboard_prints_no_bare_score_decimal_on_its_rows_chips_or_popups(tmp_path):
    TD = importlib.import_module("test_dashboard_metric_semantics")
    GF = importlib.import_module("test_dashboard_gain_of_function")
    extra = dict(GF.BLOCKS, **{
        "hself": {"slots_filled": 6400, "slots_distinct": 6400, "slots_twinned": 0, "l2_sig": 3, "l3_sig": -1,
                  "l3_steps_to_first_evidence": 2500000, "moments_with_a_slot": 0.0, "best_slot_cos": 0.2367},
        "entity_self": {"names": ["caregiver", "toy_ball"], "teacher": "wellness", "presence_mean": [0.8, 0.3],
                        "l2_link": [0.42, -0.05], "l2_ticks": 7, "windows_needed": 11, "l2_counted_names": [],
                        "l2_window_moments": 10000,
                        "reader": {"gate_open": True, "applied_weight": 0.25, "why": "open"}},
        "agency": {"pay": 0.0224, "prior_w": 0.0434, "motion": 0.44, "fovea": 0.03, "vis_adv": 0.0019,
                   "touch_adv": 0.0146},
        "plan_license": {"win_emas": [0.38, 0.71, 0.88], "licensed": 0, "mode": "earned"},
        "td_gate": {"vision": 0.4454, "tactile": 0.35},
        "tenure_pe": {"pose": 8.4e-05}, "tenure_gap": {"pose": -3.6e-09},
        "tenure_tok": {"rank1": 0.206, "n_decoys": 14.0, "vis_gate_open": 0.0}, "tenure_events": 20,
        "contact_turnover": {"born": 0.4, "died": 1.9}, "task_head_tilt": -0.35, "task_squeaker_view": 110.0,
        "task_squeaker_seen": 0.0, "task_headlook": 3, "intention": {"plan_length": 1, "final_mse": 0.8,
                                                                     "gap_closed_pct": 3.0}})
    h = TD._render(tmp_path, args={"agency_weight": 0.1, "agency_prior": 0.05, "d_model": 192}, extra=extra)
    body, pops = page_text(h), _popups(h)
    assert pops, "the popup table is embedded"
    for where, t in (("page", body), ("popups", pops)):
        b = bare_decimals(t)
        assert not b, f"old dashboard {where}: {b[:5]}"
        assert not long_numbers(t), where
        # verify-fix-dashboards 2026-09-23: the popups still said "barely", "healthy" (7 times),
        # "slightly" and "roughly" (the style guide writing rules)
        assert not banned_words(t), f"old dashboard {where}: banned wording {banned_words(t)}"
        # fix-readout-dashboards 2026-09-23: eight popup multipliers named nothing ("48x", "12x")
        assert not unplaceable(t), f"old dashboard {where}: {unplaceable(t)[:5]}"
    assert "Agency drive payments" in body and "22% of its weight's most" in body
    assert "picks its own command's effect out of a lineup of 15 in 21 of every 100 lineups" in body


def test_the_per_body_part_readability_chip_is_red_when_a_part_is_unread():
    """M9."""
    from mimo_bridge.tools import dashboard
    ph = {"per_handle": [1e-6, 1e-2], "share": [1e-6, 1e-2], "line": 2.9e-5, "below": ["gaze"], "names": ["gaze", "wrist_R"]}
    chips = dashboard.chips_for({"action_read": {"readable": True, "diff_action": 7.8, "pos_block": 1.0, "per_handle": ph}})
    assert 'class="chip b' in chips and "action UNREAD for 1 of 2 body parts" in chips and "eyes (the gaze command)" in chips
    assert "READABLE" not in chips
    chips = dashboard.chips_for({"action_read": {"readable": True, "diff_action": 7.8, "pos_block": 1.0, "handles": 26}})
    assert "all 26 handles changed at once; per-body-part readability unmeasured" in chips
    assert not bare_decimals(page_text(chips))


def test_the_lifetime_page_prints_no_bare_score_decimal_and_rebuilds_on_a_new_marker(tmp_path, monkeypatch):
    """m1: the marker changed with the content, so a copy written by older code is rebuilt at once."""
    from mimo_bridge.tools import lifetime
    GF = importlib.import_module("test_dashboard_gain_of_function")
    saved = tmp_path / "Saved" / "mimo_bridge"
    parent = GF._run(saved, "P", [GF._rec(100 + i, timestamp=1_790_000_000 + i, prediction_audit=audit(),
                                          cmd_response_cos_held=0.1 + 0.01 * i, task_hand_fov=0.02,
                                          rest_over_bout_speed=0.88, next_pe={"vision": 0.01 * (1 + i)})
                                  for i in range(10)])
    child = GF._run(saved, "P_seg2", [GF._rec(200 + i, timestamp=1_790_001_000 + i,
                                              causal_chain={"chosen_best_share": 0.5, "alternatives_n": 4},
                                              cmd_response_cos_held=0.2) for i in range(10)],
                    {"resume": str(parent / "checkpoints" / "ckpt_step_109")})
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(child), "timestamp": 1}), encoding="utf-8")
    for name, value in (("SAVED", str(saved)), ("EXP", str(saved / "experiments")),
                        ("OUT_HTML", str(saved / "lifetime.html")), ("OUT_JSON", str(saved / "lifetime.json"))):
        monkeypatch.setattr(lifetime, name, value)
    (saved / "lifetime.html").write_text('<main data-version="lineage-history-v2">old</main>', encoding="utf-8")
    lifetime.build(max_age_s=10 ** 9)
    page = (saved / "lifetime.html").read_text(encoding="utf-8")
    assert "lineage-history-v8" in page, "a page written with the old marker is rebuilt at once"
    assert not bare_decimals(page_text(page)), bare_decimals(page_text(page))[:5]
    assert not long_numbers(page_text(page))
    assert not banned_words(page_text(page)), banned_words(page_text(page))
    assert not unplaceable(page_text(page)), unplaceable(page_text(page))[:5]
    # verify-fix-dashboards: the item-1 yardstick prints its control (chance among the candidates)
    # beside it, never only its place between knowing nothing and the top of the scale (B1)
    assert "Its control in the same run: chance of choosing the best-predicted plan by luck" not in page_text(page)
    assert ls.SERIES["cc.chosen_best"]["floor_series"] is None


# --------------------------------------------------------------------------- fix-readout-dashboards
# The extended readout scanner (fix-scanner, commit 22b047f9b) found numbers a reader cannot place on
# these pages (docs/gain_of_function_v3/measurements/readout_scan_2026-09-23.txt). Each test pins one
# group of those hits at its source.
def test_the_old_dashboard_popups_say_what_every_multiplier_multiplies():
    """Eight popup multipliers named nothing: "persistence beat the model 48x", "1.16 to 2.45 times on
    the undamaged runs", "H36 1.16x", "moved body speed 12x", "clear of 1.0x", "H65 (1.40x, ...)",
    "1.4x is a floor", "slowed 3.2x at 14:16". Each now names what it multiplies or is a real unit."""
    from mimo_bridge.tools import dashboard
    texts = {f"{key}.{f}": v[f] for key, v in list(dashboard.DOC.items()) + list(dashboard.ROW_DOC.items())
             for f in ("t", "w", "h", "f")}
    for where, text in texts.items():
        assert not readout_problems(text), (where, readout_problems(text)[:3])
    joined = "\n".join(texts.values())
    for gone in ("48x", "12x", "1.0x", "1.40x", "1.4x", "1.16x", "3.2x", "2.45 times on the undamaged"):
        assert gone not in joined, gone
    assert "the model's proprioception error was 48 times the error of predicting that nothing changes" in \
        " ".join(joined.split())
    assert "1.16 to 2.45 times next-state error on the undamaged runs" in texts["chip:reconfault.w"]
    assert "H30 at 5.3 to 169 times next-state error" in texts["chip:reconfault.f"]
    assert "28 to 43 cm/s (H20-H22)" in texts["chip:gates.f"] and "3.6 to 11 cm/s (H23-H28)" in texts["chip:gates.f"]
    sit = texts["Situation dependence (controlled test).h"]
    assert "1.40 times as much as the jitter did (95% interval 1.15 to 1.72 times the jitter)" in sit
    assert "fell from 1.25 to 0.40 model steps per second at 14:16" in texts["Machine load (memory, card, step time).f"]


def test_the_liveness_cards_say_what_responding_means_without_a_raw_input_value():
    """The liveness definition printed "a unit responds when its input rises above -3": a raw input
    value of a hidden unit, a number a reader cannot place. The Now card and the Over-time card now
    say what the line means, in the explainer's and the probes' words: the flat low end of GELU, where
    a change in the input moves the output by less than 1.2% of the change."""
    known = dd.liveness_card({})
    text = " ".join(ln["text"] for ln in known["lines"])
    assert ls.RESPONDS in text and "above -3" not in text
    assert_plain(text, "Now liveness card")
    card = next(c for c in ls.CARDS if c["id"] == "liveness")
    assert ls.RESPONDS in card["what"] and "above -3" not in card["what"]
    assert not readout_problems(card["what"]), readout_problems(card["what"])
    assert "flat low end" in ls.RESPONDS and "less than 1.2% of the change" in ls.RESPONDS


# The body probe's headline as the index holds it (Saved/mimo_bridge/gof_probes/index.jsonl, line of
# 2026-09-23 03:50:45): its own summary ends with a raw hidden-unit input line and value.
BODY_PROBE_0350 = {
    "metric": "body_captured_share_tenure_cm", "value": 0.0,
    "unit": ("share of the real 250 ms change in where every body part ended up that the prediction foresaw: "
             "from none (ignored the command) to all (foresaw the whole change)"),
    "floor": 0.0, "goal": None, "distance_to_goal": None,
    "lived": ("When one limb was commanded to turn 20 degrees (finger, wrist, elbow, shoulder, knee) and the rest held "
              "still, the commanded part ended 7.6 degrees away from where holding still left it, after 250 ms; the "
              "model's 250 ms prediction of where every body part would be (its tenure pose head) foresaw 0% of the "
              "real difference the command made (0% = it ignores the command, 100% = it foresees the whole change). "
              "Distance to goal unmeasured. The tenure pose head gave the same output for every state and command: "
              "none of its 256 hidden units responded (a unit responds when its input rises above -3; the highest "
              "input any unit received was -7)."),
    "display": "0%",
    "goal_note": ("distance to goal unmeasured: no working-control reading exists for this probe yet (100% is the "
                  "scale's end, foreseeing even what no model can)"),
    "reader_liveness": {
        "live_share": 0.0, "input_min": -32.92073059082031, "input_max": -7.133721351623535, "units": 256,
        "arms": 36, "live_input_line": -3.0,
        "lived": ("Of the tenure pose head's 256 hidden units, 0% responded, averaged over the probe's 36 commands "
                  "(0% = the head returns the same number whatever the state and command, 100% = every unit "
                  "responds). A unit responds when its input rises above -3; the highest input any unit received "
                  "was -7, below that line.")}}


def _strings(value, out):
    if isinstance(value, dict):
        for v in value.values():
            _strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)
    elif isinstance(value, str):
        out.append(value)
    return out


def test_the_body_probes_own_summary_keeps_its_point_and_leaves_out_the_raw_inputs(tmp_path):
    """14 of the scan's 31 hits: the body probe's own summary, printed as written on the Now card of
    item 8 and the Over-time probe panel, and carried whole in the /api/lineage payload. The part
    that prints raw hidden-unit inputs is left out, the point of its sentence stays, and the card says
    how many parts were left out and where the whole summary is."""
    path = r"C:\probe_results\body.json"
    probes = _probe_index(tmp_path, [
        {"probe": "body", "item": 8, "checkpoint": "x/ckpt_step_344400", "lineage_step": 344400,
         "time": "2026-09-23 03:50:45", "status": "measured", "path": path, "headline": BODY_PROBE_0350}])
    entry = probes["items"]["8"][0]
    own = entry["words"]["own"]
    assert "none of its 256 hidden units responded" in own and "7.6 degrees" in own
    assert "above -3" not in own and "was -7" not in own
    assert "(1 part left out here: it prints numbers a reader cannot place; the whole summary is in " + path in own
    for text in _strings(probes, []):
        assert not readout_problems(text), (text[:120], readout_problems(text)[:3])
    assert entry["latest"]["reader_liveness"]["lived"].startswith("Of the tenure pose head's 256 hidden units")
    assert "was -7" not in entry["latest"]["reader_liveness"]["lived"]
    row = {r["item"]: r for r in ls.gain_of_function({}, {}, probes)["items"]}[8]
    text = ls.item_text(row)
    assert "The probe's own summary: When one limb was commanded to turn 20 degrees" in text
    assert_plain(text, "item 8")
    # a summary with nothing printable is withheld whole and the card says where to read it
    bare = dict(BODY_PROBE_0350, lived="The highest input any unit received was -7.")
    w = ls.probe_words(dict(bare, path=path), "body")
    assert w["own"] is None and w["own_withheld"] is True


# --------------------------------------------------------------------------- the page in a browser
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


@pytest.mark.skipif(not os.path.isfile(EDGE), reason="headless Edge is not installed here")
def test_the_page_in_headless_edge_has_no_script_error_and_no_bare_decimal(tmp_path):
    """The page as a browser renders it, on a fixture lineage served by this test's own server
    thread (never the running dashboard server): every text node of Now, Over time (every card and
    every collapsed part opened) and Live."""
    from http.server import ThreadingHTTPServer
    _lineage(tmp_path, child_args={"causal_chain": "on"}, records=rich_record())
    _write(tmp_path / "docs/development_dashboard/evidence.json", {"schema": 1, "abilities": [], "records": []})
    probe = """<script>window.__e=[];addEventListener('error',e=>__e.push(e.message));
function T(n){const o=[];if(!n)return o;const w=document.createTreeWalker(n,NodeFilter.SHOW_TEXT);let t;
while((t=w.nextNode())){const p=t.parentNode;if(!p||p.nodeName==='SCRIPT'||p.nodeName==='STYLE'||p.closest('code'))continue;
const s=(t.nodeValue||'').trim();if(s)o.push(s);}return o.join(' ');}
addEventListener('load',()=>setTimeout(()=>{
const R={e:__e,initialWords:document.getElementById('now').innerText.trim().split(/\\s+/).length,
 compact:!document.getElementById('gof-overview').open&&!document.getElementById('now-evidence').open,
 healthTitle:document.getElementById('liveness-title').textContent};
for(const d of document.querySelectorAll('details'))d.open=true;
renderGof();renderLiveness();
R.keepsOpen=[...document.querySelectorAll('#gof-items details,#liveness-lines details')].every(d=>d.open);
R.now=T(document.getElementById('now'));R.cards=[...document.querySelectorAll('.gof-item')].map(a=>T(a));
showPage('history');setTimeout(()=>{for(const d of document.querySelectorAll('details'))d.open=true;
setTimeout(()=>{R.history=T(document.getElementById('history'));R.gauges=[...document.querySelectorAll('.gauge-card')].map(a=>T(a));
showPage('live');R.live=T(document.getElementById('live'));const p=document.createElement('pre');p.id='R';p.textContent=JSON.stringify(R);
document.body.appendChild(p);},6000);},4000);},3000));</script>"""

    class H(dd.Handler):
        def do_GET(self, head=False):
            if self.path.split("?")[0].split("#")[0] == "/":
                page = (dd.ASSETS / "development_dashboard.html").read_text(encoding="utf-8")
                data = page.replace("<body>", "<body>" + probe, 1).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET(head)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    server.root = tmp_path
    threading.Thread(target=server.serve_forever, daemon=True).start()
    work = tempfile.mkdtemp(prefix="dash_test_edge_")
    try:
        proc = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                               "--disable-extensions", "--user-data-dir=" + os.path.join(work, "p"),
                               "--window-size=1280,2000", "--virtual-time-budget=30000", "--dump-dom",
                               f"http://127.0.0.1:{server.server_address[1]}/#now"], capture_output=True, timeout=240)
    finally:
        server.shutdown()
        shutil.rmtree(work, ignore_errors=True)
    m = re.search(r'<pre id="R">(.*?)</pre>', proc.stdout.decode("utf-8", "replace"), re.S)
    assert m, "the page did not finish rendering"
    R = json.loads(_html.unescape(m.group(1)))
    assert R["e"] == [], R["e"]
    assert R["compact"] and R["keepsOpen"], R
    assert R["initialWords"] < 800, R["initialWords"]
    assert "stopped responding" not in R["healthTitle"]
    assert len(R["cards"]) == 10 and len(R["gauges"]) >= 12
    for where in ("now", "history", "live"):
        assert not bare_decimals(R[where]), (where, bare_decimals(R[where])[:5])
        assert not unplaceable(R[where]), (where, unplaceable(R[where])[:5])
        assert not long_numbers(R[where]), where
        for stray in ("undefined", "NaN", "[object Object]"):
            assert stray not in R[where], (where, stray)
    for card in R["cards"] + R["gauges"]:
        assert not doubled_sentences(card), (card[:80], doubled_sentences(card)[:2])
