# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The causal chain (docs/gain_of_function_v3/DESIGN.md D13, --causal-chain on).

The recorder's pairing rules are pinned on a fake model whose predictions are known exactly:
a prediction is graded one moment later and at the order's end against the tokens that then
arrive, beside persistence; it is never paired across a reset; a planning event is stamped with
whether it reached the body. The model's half is pinned on a small real model and body: the
chain changes no order and draws from no random stream the run uses."""
import json
import os

import numpy as np
import pytest
import torch

from lem.model.wellness import APPROXIMATE_TERMS
from mimo_bridge.causal_chain import CausalChainRecorder, outcome, plan_gaps, rank_of_chosen

SENSES = ("vision", "tactile", "intermind", "motor")


class FakeModel:
    """Tokens are d-wide constants per body. The prediction for depth k is now + k (so a world
    that moves by exactly 1 per moment makes the prediction exact), the approximate evaluator
    reads 0.5 on every term now and 0.6 on every term at every predicted depth."""
    GAZE_IDX = 2

    def __init__(self, B=2, d=4, H=4):
        self.B, self.d, self.H = B, d, H
        self._transition_epoch = 0
        self._last_source_mask = {"vision": "real", "tactile": "real", "intermind": "synthetic",
                                  "motor": "real"}
        self.set_tokens(0.0)
        self.set_wellness(0.5)
        self.last_plan_record = None
        self.last_intention = None
        self.depths_asked = []

    def set_tokens(self, v):
        self.last_tokens = {m: torch.full((self.B, self.d), float(v)) for m in SENSES}

    def set_wellness(self, v):
        self.last_wellness_grounded = {n: torch.full((self.B, 1), float(v)) for n in APPROXIMATE_TERMS}

    def predict_before_execution(self, send_motor, gaze_out, depth=1):
        self.depths_asked.append(depth)
        now = {m: v.clone() for m, v in self.last_tokens.items() if m != "motor"}
        out = {}
        for k in sorted({1, depth}):
            out[k] = {"tokens": {m: v + k for m, v in now.items()},
                      "judge": torch.full((self.B, 5), 0.6)}
        return {"depths": out, "tokens_now": now, "judge_now": torch.full((self.B, 5), 0.5),
                "epoch": self._transition_epoch, "B": self.B}


def _send(B=2, H=4):
    return np.zeros((B, H, 9), np.float32)


def test_rank_counts_only_alternatives_predicted_strictly_higher():
    chosen = torch.tensor([1.0, 2.0, 3.0])
    alts = torch.tensor([[0.5, 2.5, 3.0],
                         [0.9, 2.1, 1.0],
                         [1.0, 3.0, 0.0]])
    assert rank_of_chosen(chosen, alts).tolist() == [1, 4, 1]
    # with a tie band, only an alternative higher by MORE than the band counts
    assert rank_of_chosen(chosen, alts, tie_band=0.2).tolist() == [1, 3, 1]


def test_gaps_spread_and_the_outcome_against_the_tie_band():
    """Review MAJOR 3: a gap smaller than the evaluator's resolution is a tie, never 'best'."""
    pw = {"chosen": [1.0, 2.0, 0.5], "hold": [0.99, 2.5, 0.1], "reflex": [0.9, 1.0, 0.2]}
    g = plan_gaps(pw, ["hold", "reflex"])
    assert g["gap"]["hold"] == [0.01, -0.5, 0.4]
    assert g["gap_to_best_alt"] == [0.01, -0.5, 0.3]
    assert g["candidate_spread"] == [0.1, 1.5, 0.4]
    assert [outcome(v, 0.05) for v in g["gap_to_best_alt"]] == ["tied", "behind", "best"]
    assert outcome(0.3, None) is None


def test_the_tie_band_is_the_measured_one():
    """Review MAJOR 3: the band a recorder uses by default is recomputed here from every twin the
    measurement kept (24 moments x 5 orders on ckpt_step_344400): the 90th percentile of Judge B's
    error on the difference between two orders from the same moment."""
    import itertools
    from pathlib import Path
    import mimo_bridge.causal_chain as CC
    src = Path(__file__).resolve().parents[4] / CC.TIE_BAND_MEASUREMENT["source"]
    data = json.loads(src.read_text(encoding="utf-8"))
    errs, agree = [], 0
    for r in data["rows"]:
        for a, b in itertools.combinations(sorted(r["pred"]), 2):
            pd = r["pred"][a] - r["pred"][b]
            rd = r["real"][a] - r["real"][b]
            errs.append(abs(pd - rd))
            agree += int(np.sign(pd) == np.sign(rd))
    assert len(errs) == 240 and len(data["rows"]) == 24
    assert CC.TIE_BAND == round(float(np.quantile(errs, 0.9)), 4)
    assert CC.TIE_BAND_MEASUREMENT["sign_agreement_pairs"] == [agree, len(errs)]
    assert CausalChainRecorder(None, hold=5).window_summary()["tie_band"] == CC.TIE_BAND


def test_a_latched_order_is_graded_one_moment_later_and_at_its_end(tmp_path):
    m = FakeModel()
    r = CausalChainRecorder(str(tmp_path), hold=3, strict=True)
    r.on_latch(m, 10, latched=True, order_seq=7, send_motor=_send(), gaze_out=None, source="reflex")
    assert m.depths_asked == [3] and len(r.pending) == 1
    m.set_tokens(1.0)                  # the world moved by exactly the predicted 1
    r.after_step(m, 11)
    r.after_step(m, 12)
    m.set_tokens(2.5)                  # by the order's end it moved 2.5, not the predicted 3
    m.set_wellness(0.6)
    r.after_step(m, 13)
    assert not r.pending
    w = r.window_summary()
    assert w["events"] == 1 and w["latch_predictions"] == 1 and w["planner_events"] == 0
    # intermind is a synthetic placeholder: never graded; motor is not a predicted sense
    assert w["pred_err_next"] == {"vision": 0.0, "tactile": 0.0}
    assert w["persist_err_next"] == {"vision": 1.0, "tactile": 1.0}
    assert w["pred_err_latch"] == {"vision": 0.25, "tactile": 0.25}
    assert w["persist_err_latch"] == {"vision": 6.25, "tactile": 6.25}
    # the approximate evaluator predicted +0.1 on each of 5 terms; the grounded five rose 0.1 each
    assert abs(w["predicted_wellness_change"] - 0.5) < 1e-6
    assert abs(w["realized_wellness_change"] - 0.5) < 1e-6
    assert w["wellness_change_corr"] is None, "no spread across 2 identical bodies: no correlation"
    lines = (tmp_path / "causal_chain.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["type"] == "latch" and rec["order_seq"] == 7 and rec["source"] == "reflex"
    assert rec["next"]["pred_err"]["vision"] == 0.0 and rec["latch"]["persist_err"]["tactile"] == 6.25
    assert len(lines[0]) < 2000, "a per-event record stays about 1 KB"


def test_nothing_pairs_across_a_reset_or_a_teleport(tmp_path):
    m = FakeModel()
    r = CausalChainRecorder(str(tmp_path), hold=3, strict=True)
    r.on_latch(m, 20, latched=True, order_seq=1, send_motor=_send(), gaze_out=None, source="reflex")
    m._transition_epoch += 1           # the model reset between the prediction and its moment
    r.after_step(m, 21)
    r.on_latch(m, 21, latched=True, order_seq=2, send_motor=_send(), gaze_out=None, source="reflex")
    r.clear("scheduled episode reset")
    for s in range(22, 26):
        r.after_step(m, s)
    w = r.window_summary()
    assert w["events"] == 0 and w["dropped_at_reset"] == 2
    assert w["pred_err_next"] is None
    assert not (tmp_path / "causal_chain.jsonl").exists()


def _plan(step, B=2, H=4):
    exec_form = torch.zeros(B, H, 9)
    exec_form[..., 0] = 10.0
    exec_form[..., 6] = 1.0
    return {"step": step, "bodies": B,
            "pred_tokens": {"vision": torch.full((B, 4), 1.0), "tactile": torch.full((B, 4), 1.0)},
            "exec_form": exec_form, "reach_clamp_removed_share": [0.0, 0.5],
            "pred_wellness": {"chosen": [1.0, 2.0], "hold": [0.5, 2.5], "reflex": [0.9, 2.1],
                              "perturb_1": [1.0, 3.0], "perturb_2": [0.1, 0.2]},
            "goal_distance": {"chosen": [0.1, 0.9], "hold": [0.2, 0.5], "reflex": [0.3, 1.0],
                              "perturb_1": [0.4, 1.1], "perturb_2": [0.5, 1.2]},
            "chosen_rank": [1, 3], "alternatives": ["hold", "reflex", "perturb_1", "perturb_2"]}


def test_a_planning_event_is_stamped_with_whether_it_reached_the_body(tmp_path):
    m = FakeModel(H=4)
    r = CausalChainRecorder(str(tmp_path), hold=2, strict=True, tie_band=0.05)
    # a plan made on a step that is not a latch: computed, never sent
    m.last_plan_record = _plan(30)
    m.last_intention = {"step": 30, "reached_body": None}
    r.on_latch(m, 30, latched=False, order_seq=4, send_motor=_send(), gaze_out=None, source="planner")
    assert m.last_intention["reached_body"] is False and not r.pending
    # a plan latched this step; the body was sent an order 3 cm (x) from the one it imagined on
    # body 0 and exactly the imagined one on body 1 (the eyes' row, index 2, is left out)
    m.last_plan_record = _plan(40)
    m.last_intention = {"step": 40, "reached_body": None}
    sent = m.last_plan_record["exec_form"].clone()
    sent[0, :, 0] += 3.0
    sent[0, m.GAZE_IDX, 0] += 100.0
    r.on_latch(m, 40, latched=True, order_seq=5, send_motor=sent.numpy(), gaze_out=None,
               source="planner")
    assert m.last_intention["reached_body"] is True and m.last_intention["order_seq"] == 5
    assert abs(m.last_intention["plan_exec_distance_cm"] - 1.5) < 1e-6
    assert "plan_share_of_executed" not in m.last_intention
    m.set_tokens(1.0)
    r.after_step(m, 41)
    r.after_step(m, 42)
    w = r.window_summary()
    assert w["planner_events"] == 2 and w["reached_body_share"] == 0.5
    assert abs(w["plan_exec_distance_cm"] - 1.5) < 1e-6 and w["plan_exec_turn_deg"] == 0.0
    assert "planner_share_of_executed" not in w
    assert abs(w["reach_clamp_removed_share"] - 0.25) < 1e-9
    # strict ranks 1 and 3 over two plans x two bodies; against the tie band (0.05) body 0's
    # chosen plan TIED the best alternative (1.0 against 1.0) and body 1's fell behind (2.0 v 3.0)
    assert w["chosen_rank_mean"] == 2.0 and w["chosen_rank_resolved_mean"] == 2.5
    assert w["tie_band"] == 0.05
    assert w["chosen_best_share"] == 0.0 and w["chosen_tied_share"] == 0.5
    assert w["chosen_behind_share"] == 0.5
    assert abs(w["gap_to_best_alt_mean"] - (-0.5)) < 1e-9
    assert abs(w["candidate_spread_mean"] - 1.85) < 1e-9
    assert w["alternatives_n"] == 4
    # best alternative per body: 1.0 and 3.0 (twice each)
    assert abs(w["pred_wellness_chosen"] - 1.5) < 1e-9 and abs(w["pred_wellness_best_alt"] - 2.0) < 1e-9
    # the planner's own objective: chosen closest on body 0, not on body 1
    assert w["chosen_closest_share"] == 0.5
    # the acting route's depth-1 tokens (1.0) matched the moment that came (1.0)
    assert w["plan_pred_err_next"] == {"vision": 0.0, "tactile": 0.0}
    recs = [json.loads(x) for x in (tmp_path / "causal_chain.jsonl").read_text().splitlines()]
    assert [x["type"] for x in recs] == ["plan", "latch"]
    assert recs[0]["reached_body"] is False and recs[1]["reached_body"] is True
    assert "pred_tokens" not in recs[1]["plan"], "tensors never reach the JSON record"
    assert "exec_form" not in recs[1]["plan"]
    gaps = recs[1]["plan"]["pred_wellness_gaps"]
    assert gaps["outcome"] == ["tied", "behind"] and gaps["tie_band"] == 0.05
    assert recs[1]["plan_exec_distance"] == {"cm": [3.0, 0.0], "deg": [0.0, 0.0]}


def test_the_plan_to_executed_distance_is_taken_against_the_order_the_worlds_recorded(tmp_path):
    """Review of the limb fixes (2026-09-25, finding 4): with the calf, chest or arm fix on, the worlds
    change the order before it acts, so a distance taken against the order as sent would keep reading
    'the body got what was imagined'. The trainer hands over the order the worlds recorded after the
    burst, and the window, the planning record and the event's line all measure against that."""
    m = FakeModel(H=4)
    r = CausalChainRecorder(str(tmp_path), hold=2, strict=True)
    m.last_plan_record = _plan(40)
    m.last_intention = {"step": 40, "reached_body": None}
    sent = m.last_plan_record["exec_form"].clone()            # exactly what the planner imagined
    r.on_latch(m, 40, latched=True, order_seq=5, send_motor=sent.numpy(), gaze_out=None, source="planner")
    assert m.last_intention["plan_exec_distance_cm"] == 0.0
    recorded = sent.clone()
    recorded[:, 0, 0] += 4.0                                  # the worlds moved row 0 by 4 cm on both bodies
    recorded[:, m.GAZE_IDX, 0] += 100.0                       # the eyes' row is left out, as before
    r.on_executed(m, 40, recorded.numpy())
    # 4 cm on 1 of 3 body rows (the record keeps 3 decimals)
    assert abs(m.last_intention["plan_exec_distance_cm"] - 4.0 / 3.0) < 1e-3
    r.on_executed(m, 40, sent.numpy())                        # a second hand-over changes nothing
    assert abs(m.last_intention["plan_exec_distance_cm"] - 4.0 / 3.0) < 1e-3
    m.set_tokens(1.0)
    r.after_step(m, 41)
    r.after_step(m, 42)
    w = r.window_summary()
    assert abs(w["plan_exec_distance_cm"] - 4.0 / 3.0) < 1e-3 and w["plan_exec_against_ledger"] == 1
    rec = [json.loads(x) for x in (tmp_path / "causal_chain.jsonl").read_text().splitlines()][-1]
    assert rec["plan_exec_distance"]["cm"] == [pytest.approx(4.0 / 3.0, abs=1e-4)] * 2
    assert "worlds recorded" in rec["plan_exec_distance"]["against"]
    # an executed order for another step, or after the next latch, revises nothing
    m.last_plan_record = _plan(50)
    m.last_intention = {"step": 50, "reached_body": None}
    r.on_latch(m, 50, latched=True, order_seq=6, send_motor=sent.numpy(), gaze_out=None, source="planner")
    r.on_executed(m, 49, recorded.numpy())
    r.on_latch(m, 55, latched=True, order_seq=7, send_motor=sent.numpy(), gaze_out=None, source="reflex")
    r.on_executed(m, 50, recorded.numpy())
    assert m.last_intention["plan_exec_distance_cm"] == 0.0
    w = r.window_summary()
    assert w["plan_exec_distance_cm"] == 0.0 and w["plan_exec_against_ledger"] == 0


def test_with_the_limb_fixes_off_nothing_is_handed_over_and_the_block_reads_as_before(tmp_path):
    """Flags off, the trainer never calls on_executed: the distance is the sent order's, as before."""
    m = FakeModel(H=4)
    r = CausalChainRecorder(str(tmp_path), hold=2, strict=True)
    m.last_plan_record = _plan(40)
    m.last_intention = {"step": 40, "reached_body": None}
    sent = m.last_plan_record["exec_form"].clone()
    sent[0, :, 0] += 3.0
    r.on_latch(m, 40, latched=True, order_seq=5, send_motor=sent.numpy(), gaze_out=None, source="planner")
    w = r.window_summary()
    assert abs(w["plan_exec_distance_cm"] - 1.5) < 1e-6 and w["plan_exec_against_ledger"] == 0


def test_the_order_end_forecast_is_also_split_by_the_orders_author(tmp_path):
    """The planner's own orders read apart (review finding 5): the pooled readings are unchanged."""
    m = FakeModel()
    r = CausalChainRecorder(str(tmp_path), hold=2, strict=True)
    r.on_latch(m, 0, latched=True, order_seq=1, send_motor=_send(), gaze_out=None, source="planner")
    r.after_step(m, 1)
    m.set_tokens(2.0)                  # predicted 2 at the order's end: exact
    r.after_step(m, 2)
    r.on_latch(m, 2, latched=True, order_seq=2, send_motor=_send(), gaze_out=None, source="reflex_noise")
    r.after_step(m, 3)
    m.set_tokens(3.0)                  # predicted 4 (2 + 2): one unit off; persistence one unit off
    r.after_step(m, 4)
    w = r.window_summary()
    assert w["pred_err_latch_by_source"]["planner"] == {"vision": 0.0, "tactile": 0.0}
    assert w["persist_err_latch_by_source"]["planner"] == {"vision": 4.0, "tactile": 4.0}
    assert w["pred_err_latch_by_source"]["reflex_noise"] == {"vision": 1.0, "tactile": 1.0}
    assert w["pred_err_latch"] == {"vision": 0.5, "tactile": 0.5}
    assert r.window_summary()["pred_err_latch_by_source"] is None, "a fresh window starts empty"


def test_the_order_end_carries_each_terms_predicted_and_realised_change(tmp_path):
    """b3-wellness: per term of the approximate evaluator, what it predicted the order would do
    to that term against what the grounded term did, at the order's end only."""
    m = FakeModel()
    r = CausalChainRecorder(str(tmp_path), hold=2, strict=True)
    r.on_latch(m, 0, latched=True, order_seq=1, send_motor=_send(), gaze_out=None, source="reflex")
    r.after_step(m, 1)
    m.set_wellness(0.7)
    r.after_step(m, 2)
    rec = json.loads((tmp_path / "causal_chain.jsonl").read_text().splitlines()[-1])
    assert "wellness_terms_pred_real" not in rec["next"]
    for n in APPROXIMATE_TERMS:
        assert rec["latch"]["wellness_terms_pred_real"][n] == [pytest.approx(0.1), pytest.approx(0.2)]
    w = r.window_summary()
    t = w["wellness_terms_latch"]["capability"]
    assert t["pred"] == pytest.approx(0.1) and t["real"] == pytest.approx(0.2) and t["corr"] is None


def test_a_rename_a_reader_blocks_is_retried_and_never_drops_the_line(tmp_path, monkeypatch):
    """Review MINOR 7: os.replace fails on Windows while a reader holds the file. The rename is
    retried; if it still fails the line is appended to the current file and counted."""
    import mimo_bridge.causal_chain as CC
    m = FakeModel()
    real = os.replace
    fails = {"n": 2}

    def flaky(a, b):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise PermissionError(5, "Access is denied")
        return real(a, b)
    monkeypatch.setattr(CC.os, "replace", flaky)
    r = CausalChainRecorder(str(tmp_path), hold=1, max_bytes=1000, strict=True,
                            rotate_waits=(0.0, 0.0, 0.0))
    lines_written = 0
    for s in range(0, 8, 2):
        r.on_latch(m, s, latched=True, order_seq=s, send_motor=_send(), gaze_out=None, source="reflex")
        r.after_step(m, s + 1)
        lines_written += 1
    names = sorted(os.listdir(tmp_path))
    assert len(names) >= 2, "the rename succeeded on a retry"
    total = sum(len((tmp_path / n).read_text().splitlines()) for n in names)
    assert total == lines_written and r.window_summary()["rotation_failures"] == 0
    # a reader that never lets go: the rename gives up, the line is kept, the failure counted
    fails["n"] = 10 ** 6
    r.on_latch(m, 20, latched=True, order_seq=20, send_motor=_send(), gaze_out=None, source="reflex")
    r.after_step(m, 21)
    total2 = sum(len((tmp_path / n).read_text().splitlines()) for n in sorted(os.listdir(tmp_path)))
    assert total2 == lines_written + 1
    assert r.window_summary()["rotation_failures"] >= 1


def test_the_file_is_renamed_aside_past_its_limit(tmp_path):
    m = FakeModel()
    # the limit must hold at least one line: a latch line of this fake is about 730 bytes since
    # the per-term wellness change joined the order's-end grade (b3-wellness, 2026-09-22; it was
    # about 560 before), so the limit moved from 600 to 1,000 and still forces a rename
    r = CausalChainRecorder(str(tmp_path), hold=1, max_bytes=1000, strict=True)
    for s in range(0, 12, 2):
        r.on_latch(m, s, latched=True, order_seq=s, send_motor=_send(), gaze_out=None, source="reflex")
        r.after_step(m, s + 1)
    names = sorted(os.listdir(tmp_path))
    assert "causal_chain.jsonl" in names and len(names) >= 2
    assert all(os.path.getsize(tmp_path / n) <= 1000 for n in names)


def test_a_failure_is_counted_not_raised_in_a_run(tmp_path):
    class Broken(FakeModel):
        def predict_before_execution(self, *a, **k):
            raise RuntimeError("boom")
    r = CausalChainRecorder(str(tmp_path), hold=2)
    r.on_latch(Broken(), 1, latched=True, order_seq=1, send_motor=_send(), gaze_out=None, source="reflex")
    assert r.window_summary()["failures"] == 1


# ---------------------------------------------------------------- the model's half, on a body
def _rng():
    return torch.get_rng_state().clone(), np.random.get_state()


def _set_rng(s):
    torch.set_rng_state(s[0])
    np.random.set_state(s[1])


def _same_np_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


@pytest.mark.slow
def test_the_chain_changes_no_order_and_draws_from_no_stream_the_run_uses(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)

    def build(causal):
        return LemLite(device="cpu", seed=0, intention=True, intention_every=5,
                       imagination=False, no_vision=True, next_frame_loss=True,
                       smooth_actions=0.35, causal_chain=causal,
                       overrides={"training": {"steps_per_phase": 3000}})
    a, b = build(False), build(True)
    rec = CausalChainRecorder(str(tmp_path), hold=5, strict=True)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "cc"), tag="_cc", vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(a.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        prev, prev_id, held, seq, plans, checks = None, None, None, 0, [], []
        for i in range(27):
            if i in (20, 25):
                # This tests recorder neutrality when an action is planned. Pin the
                # same earlier goal in both learners: a changed exploration recipe
                # may otherwise make the present moment win and correctly produce
                # no plan (step 20 had exactly zero goal error after the quaternion
                # composition repair). Goal ranking is covered separately.
                for model in (a, b):
                    remembered, value = model.goal_ring[0]
                    model.goal_ring[0] = (remembered, torch.full_like(value, 1e6))
            s0 = _rng()
            ra = a.step(frames, dev, train=True, step_index=i, executed_action=prev,
                        executed_order_id=prev_id)
            sa = _rng()
            _set_rng(s0)
            rb = b.step(frames, dev, train=True, step_index=i, executed_action=prev,
                        executed_order_id=prev_id)
            rec.after_step(b, i)
            latched = i % 5 == 0 or held is None
            if latched:
                held, seq = ra.motor_for_action, seq + 1
            rec.on_latch(b, i, latched=latched, order_seq=seq, send_motor=held,
                         gaze_out=ra.gaze_out, source=b.last_action_source)
            sb = _rng()
            assert torch.equal(sa[0], sb[0]) and _same_np_state(sa[1], sb[1]), \
                f"step {i}: the causal chain moved a global random stream"
            assert torch.equal(a.noise_rng.get_state(), b.noise_rng.get_state())
            assert np.array_equal(ra.motor_for_action, rb.motor_for_action), f"step {i}: order changed"
            assert np.array_equal(ra.motor_out, rb.motor_out)
            assert ra.loss == rb.loss, f"step {i}: the loss changed"
            if b.last_plan_record is not None and b.last_plan_record.get("step") == i:
                assert any(not torch.equal(b._last_plan_inputs['cur'][m],
                                           b._last_plan_inputs['goal'][m])
                           for m in b.active if m != 'motor')
                plans.append(b.last_plan_record)
                checks.append(_route_and_record_checks(b, build))
            elif i % 5 != 0:
                assert b.last_plan_record is None, "a planning record outlived its step"
            frames = src.step(held, ra.gaze_out, None, render=False)
            prev, prev_id = held, seq
        for pa, pb in zip(a.params, b.params):
            assert torch.equal(pa, pb), "the causal chain changed a weight"
    finally:
        src.close()
    assert [p["step"] for p in plans] == [20, 25]
    for p in plans:
        assert p["alternatives"] == ["hold", "reflex", "perturb_1", "perturb_2"]
        assert set(p["pred_wellness"]) == {"chosen", "hold", "reflex", "perturb_1", "perturb_2"}
        # THE RANK IS THE RANK OF THE RECORDED SUMS (review MINOR 9: the old pin was 1 <= rank <= 5)
        sums = torch.tensor([p["pred_wellness"][n] for n in ["chosen"] + p["alternatives"]])
        assert p["chosen_rank"] == rank_of_chosen(sums[0], sums[1:]).tolist()
        assert p["goal"]["age_steps"][0] >= 0 and p["goal"]["ring_size"] >= 20
        assert set(p["pred_tokens"]) >= {"proprioception", "tactile"}
        assert p["exec_form"].shape == (1, len(b.handle_names), 9)
        assert 0.0 <= p["reach_clamp_removed_share"][0] <= 1.0
    for c in checks:
        # the chosen plan scored through the alternatives' route gives the planner's own sum
        assert c["route_gap"] < 1e-5, c
        assert c["replayed_same_plan"], c
        assert c["record_round_trip"], c
    w = rec.window_summary()
    assert w["latch_predictions"] >= 5 and w["events"] >= 4 and w["failures"] == 0
    assert w["planner_events"] == 2 and w["reached_body_share"] == 1.0
    assert set(w["pred_err_next"]) >= {"proprioception", "tactile"}
    # the order sent is the order the planner imagined sending (the smoother included in both)
    assert w["plan_exec_distance_cm"] < 1e-3 and w["plan_exec_turn_deg"] < 1e-2
    assert "planner_share_of_executed" not in w
    assert w["candidate_spread_mean"] is not None and w["gap_to_best_alt_mean"] is not None


def _route_and_record_checks(model, build):
    """Right after a planning step: (1) the plan replayed from its captured decision is the same
    plan, and scored through the ALTERNATIVES' route (LemLite._causal_score_candidates) it gives
    the planner's own depth-1 predicted wellness (review MINOR 9: nothing pinned that the chosen
    action and its alternatives are scored on the same route); (2) the REAL planning record
    round-trips through the episode companion (capture, torch.save, load, restore)."""
    import io as _io
    from mimo_bridge import continuation_state as CS
    rec = model.last_plan_record
    plan0 = model._last_plan_inputs["plan0"]
    decision = model._last_plan_inputs["decision"]
    again, meta = model.replan_last(model.intention_gen._wellness_w)
    d1 = meta["depth1"]
    _preds, well = model._causal_score_candidates([plan0], d1["state"], decision)
    route_gap = float((well.sum(-1) - d1["well"].float().sum(-1)).abs().max())
    state = CS.capture_learner(model, include_replay=False)
    buf = _io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    other = build(True)
    CS.restore_learner(other, torch.load(buf, weights_only=False))
    got = other.last_plan_record
    same = (got is not None and set(got) == set(rec)
            and got["pred_wellness"] == rec["pred_wellness"]
            and got["chosen_rank"] == rec["chosen_rank"]
            and all(torch.equal(got["pred_tokens"][k], rec["pred_tokens"][k]) for k in rec["pred_tokens"])
            and torch.equal(got["exec_form"], rec["exec_form"]))
    return {"route_gap": route_gap, "replayed_same_plan": bool(torch.equal(again, plan0)),
            "record_round_trip": bool(same)}


def _micro_run(tmp_path, name, extra):
    """A one-body trainer micro-run in a temporary folder, the form
    test_explainer_producers._run_trainer established."""
    from mimo_bridge import train_mimo as TM
    out = str(tmp_path / name)
    os.makedirs(out, exist_ok=True)
    argv = ["--n-envs", "1", "--no-vision", "--minutes", "5", "--scene", "reach", "--out", out,
            "--video-every-min", "0", "--replay", "off", "--curriculum", "on",
            "--next-frame-loss", "on", "--intention", "on", "--intention-every", "5",
            "--report-every", "10", "--max-steps", "31"] + extra
    assert TM.main(argv) == 0, "trainer micro-run failed"
    recs = [json.loads(x) for x in open(os.path.join(out, "replay_training_stats.jsonl"),
                                        encoding="utf-8") if x.strip()]
    return out, [r for r in recs if r.get("type") == "mimo"]


@pytest.mark.slow
def test_the_trainer_writes_the_three_blocks_only_when_asked(tmp_path):
    out, mimo = _micro_run(tmp_path, "on", ["--causal-chain", "on", "--efference-heads", "on"])
    assert len(mimo) >= 3
    last = mimo[-1]
    m = last["machine"]
    assert m["rss_gb"] > 0 and m["step_ms"] > 0 and "causal_chain_ms" in m
    # the efference lesson's own time (review MINOR 6), inside model_ms
    assert 0.0 < m["efference_ms"] < m["model_ms"]
    cc = last["causal_chain"]
    assert cc["latch_predictions"] >= 1 and cc["failures"] == 0
    assert any(r["causal_chain"]["planner_events"] for r in mimo)
    eff = [r["efference"] for r in mimo]
    assert all(e["events_next"] > 0 for e in eff[1:])
    assert all(e["version"] == "efference_v2" and set(e["pool_spread"]) == {"body_pool", "gaze_pool"}
               for e in eff[1:])
    assert all(0.0 <= v <= 1.0 for e in eff[1:] for v in e["liveness"].values())
    assert all(e["loss_body_core"] is not None for e in eff[1:])
    lines = open(os.path.join(out, "causal_chain.jsonl"), encoding="utf-8").read().splitlines()
    types = {json.loads(x)["type"] for x in lines}
    assert "latch" in types
    intents = [r["intention"] for r in mimo if r.get("intention")]
    assert intents and all(i.get("reached_body") in (True, False) for i in intents)
    # the flags off: the machine block, and nothing else new
    _, mimo_off = _micro_run(tmp_path, "off", [])
    last_off = mimo_off[-1]
    assert "machine" in last_off and "causal_chain_ms" not in last_off["machine"]
    assert "efference_ms" not in last_off["machine"]
    assert "causal_chain" not in last_off and "efference" not in last_off
    assert all(r["intention"] is None or r["intention"].get("reached_body") is None
               for r in mimo_off)
