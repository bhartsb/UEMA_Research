# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/persistent_memory_repair_measurements.py, persistent_memory_records.py and
persistent_memory_card_proof.py on synthetic memories, rows and a small real LemLite (no checkpoint): the moment
sets' plumbing, the re-fit's three arms starting from the same predictions, the push readings' arithmetic (saved
moments and the lived step), the version-2 rules, the records reader on a plain and on a switched memory, and the
card proof's readings on the CPU's bf16."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

DOCS = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, DOCS / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _module("persistent_memory_repair_measurements")
R = _module("persistent_memory_records")
CP = _module("persistent_memory_card_proof")


def test_subset_and_cat_keep_every_field_aligned():
    s = dict(cur={"vision": torch.arange(4.).reshape(4, 1, 1)}, state=torch.arange(8.).reshape(4, 2),
             mask=torch.ones(4, dtype=torch.bool), body=torch.tensor([0, 1, 2, 3]),
             cluster=["a", "b", "c", "d"], graded={"vision": torch.tensor([True, False, True, True])})
    sub = M._subset(s, torch.tensor([True, False, True, False]))
    assert sub["cluster"] == ["a", "c"] and sub["body"].tolist() == [0, 2]
    assert sub["graded"]["vision"].tolist() == [True, True]
    both = M._cat([sub, M._subset(s, torch.tensor([False, True, False, True]))])
    assert both["cluster"] == ["a", "c", "b", "d"] and both["state"].shape == (4, 2)
    assert M._cat([None]) is None


def test_push_readings_name_growth_and_saturation():
    """A lesson whose shared gradient points against the constant grows it (descent adds -g); its push on
    the state through the readout is -W^T g, read against the saturation pattern."""
    n_mod, d, S = 2, 3, 4
    const = torch.ones(n_mod, d)
    W = torch.zeros(n_mod * d, S)
    W[:, 0] = 1.0
    pattern = torch.tensor([1.0, -1.0, 1.0, 1.0])
    g = -torch.ones(5, n_mod, d) + 0.1 * torch.randn(5, n_mod, d)
    g = g - g.mean(0) - torch.ones(n_mod, d)                 # shared part exactly -1
    r = M.push_readings(g, const, W, pattern, ["a", "b"])
    assert r["grow_constant_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert r["by_slot"]["a"]["grow_constant_cosine"] == pytest.approx(1.0, abs=1e-6)
    # -W^T shared = +6 on state unit 0, whose pattern sign is +1
    assert r["state_push_projection_on_saturation"] == pytest.approx(6.0 / 2.0, abs=1e-5)
    assert r["state_push_cosine_with_saturation"] == pytest.approx(0.5, abs=1e-6)
    assert r["moment_specific_rms"] > 0


def test_the_lived_step_readings_share_the_push_over_the_bodies():
    """P2's aggregation: per lesson the push every body shares (summed over the memory calls it reached) and its
    projection on the sign pattern of the bodies' mean lived state."""
    pattern = [1.0, -1.0, 1.0, 1.0]
    bodies = [dict(state=[0.999, -0.999, 0.999, 0.999],
                   lessons={"contact_head": {"delayed_lesson": [2.0, 0.0, 0.0, 0.0], "lived_moment": [0.0, -2.0, 0.0, 0.0]},
                            "efference_body": {"delayed_lesson": [-1.0, 0.0, 0.0, 0.0]}}),
              dict(state=[0.999, -0.999, 0.999, 0.999],
                   lessons={"contact_head": {"delayed_lesson": [2.0, 0.0, 0.0, 0.0], "lived_moment": [0.0, -2.0, 0.0, 0.0]},
                            "efference_body": {"delayed_lesson": [1.0, 0.0, 0.0, 0.0]}})]
    out = M.live_readings(bodies)
    c = out["lessons"]["contact_head"]
    assert c["shared_push_projection_on_saturation"] == pytest.approx((2 + 2) / 2.0, abs=1e-6)
    assert c["body_specific_rms"] == pytest.approx(0.0, abs=1e-9)
    e = out["lessons"]["efference_body"]
    assert e["shared_push_projection_on_saturation"] == pytest.approx(0.0, abs=1e-9) and e["body_specific_rms"] > 0
    assert out["pattern_share_of_state_above_0p99"] == 1.0 and len(pattern) == 4


def _row(step, *, refused=False, met=True, spread=0.2, window=0.3, a0=None, ratios=None):
    row = dict(step=step)
    if refused:
        row["switch"] = dict(status="refused")
        return row
    row["switch"] = dict(new_state_share_above_0p99=0.1, new_state_spread_between_bodies=spread,
                         new_states_spread_over_window=window, update_gate_share_above_0p99=0.01)
    lim = 2e-5
    row["exactness"] = dict(live=dict(met=met, change_max_abs=0.0 if met else 1.0, limit=lim),
                            draw=dict(met=True, change_max_abs=0.0, limit=lim),
                            needs_forecast=dict(change_max_abs=0.0, largest_abs=0.9))
    ratios = ratios or {"A": {"vision": 1.2, "tactile": 1.0}, "M": {"vision": 1.1, "tactile": 1.0},
                        "Z": {"vision": 1.3, "tactile": 1.0}}
    a0 = a0 or {"vision": 1.3, "tactile": 1.0}
    return row


def test_summary_applies_the_declared_rules():
    rows = [_row(344400, refused=True), _row(380550, refused=True), _row(399225), _row(402150)]
    att = [dict(step=347791, live=dict(lessons={
        "next_moment_vision_production_weights": dict(state_push_projection_on_saturation=3.0),
        "next_moment_tactile_production_weights": dict(state_push_projection_on_saturation=0.5),
        "next_moment_vision_unit_weights": dict(state_push_projection_on_saturation=9.0),
        "latent_penalty": dict(state_push_projection_on_saturation=-1.0)}),
        lived_step=dict(lessons={"contact_head": dict(shared_push_projection_on_saturation=5.0),
                                 "efference_body": dict(shared_push_projection_on_saturation=1.0),
                                 "manager_total": dict(shared_push_projection_on_saturation=50.0)}))]
    s = M.summarize(rows, att)
    assert s["population"] == [399225, 402150]
    assert s["EM"]["met"] and s["EM"]["declared_refusals"] == {"344400": [True], "380550": [True]}
    assert s["VM"]["met"]
    assert "RM2" not in s                                        # withdrawn after the second review
    assert s["P"]["named"] == "next_moment_vision_production_weights"    # unit weights are not named
    assert s["P2"]["named"] == "contact_head"                              # the manager's total is not a lesson
    assert not M.summarize([_row(399225, met=False)], [])["EM"]["met"]
    assert not M.summarize([_row(399225, window=0.01)], [])["VM"]["met"]
    assert not M.summarize([_row(399225, spread=0.01)], [])["VM"]["met"]
    # a refusal inside the population is listed, not counted as a miss ("where the switch does not refuse")
    r = M.summarize([_row(399225), _row(402150, refused=True)], [])["EM"]
    assert r["met"] and r["refused_in_population"] == [402150]


def test_the_requests_on_a_switched_memory_split_what_every_body_shares():
    """P2 on a switched memory: per lesson, the request on the memory bias every body shares at this moment against
    the part that differs between bodies."""
    bodies = [dict(bias_requests={"next_frame": {"delayed_lesson": [1.0, 0.0], "lived_moment": [0.0, 1.0]},
                                  "contact_head": {"delayed_lesson": [1.0, 0.0]}}),
              dict(bias_requests={"next_frame": {"delayed_lesson": [1.0, 0.0], "lived_moment": [0.0, 1.0]},
                                  "contact_head": {"delayed_lesson": [-1.0, 0.0]}})]
    out = M.request_readings(bodies)
    assert out["next_frame"]["all_calls"]["body_specific_rms"] == pytest.approx(0.0, abs=1e-9)
    assert out["next_frame"]["all_calls"]["shared_share_of_total"] == pytest.approx(1.0, abs=1e-6)
    assert out["contact_head"]["all_calls"]["shared_rms"] == pytest.approx(0.0, abs=1e-9)
    assert out["contact_head"]["all_calls"]["shared_share_of_total"] == pytest.approx(0.0, abs=1e-6)
    # per call site (version 3 tracks each site's shared request apart)
    assert out["next_frame"]["delayed_lesson"]["shared_rms"] == pytest.approx(0.5 ** 0.5, rel=1e-5)
    assert out["next_frame"]["lived_moment"]["body_specific_rms"] == pytest.approx(0.0, abs=1e-9)
    assert "lived_moment" not in out["contact_head"]

def test_unrepair_returns_the_plain_memory():
    from lem.model.persistent_memory import PersistentMemory
    pm = PersistentMemory(8, 2, 4, 3, hidden_size=6)
    keys = set(pm.state_dict())
    pm.enable_repair("centred")
    assert set(pm.state_dict()) != keys
    M.unrepair(pm)
    assert set(pm.state_dict()) == keys and pm.repair == "off"


def _saturated_weights(seed=1, H=8, d=4, weight_sd=2.0):
    from lem.model.persistent_memory import PersistentMemory
    torch.manual_seed(seed)
    F = (2 + 1) * d + 10 + 3          # the event layout build_memory reads: (senses + 1) x d + 10 + the order
    pm = PersistentMemory(F, 2, d, 3, hidden_size=H)
    p = torch.sign(torch.randn(H))
    p[p == 0] = 1
    with torch.no_grad():
        pm.recurrent.bias_ih_l0[2 * H:] = 30.0 * p
        pm.readout.weight.normal_(0, weight_sd)
        pm.readout.bias.normal_(0, 0.5)
    w = {k: v.clone() for k, v in pm.state_dict().items()}
    B, T = 4, 20
    ev = torch.randn(F) + 0.3 * torch.randn(B, T, F)
    t = (torch.arange(T).float() * 0.5 + 10).expand(B, -1).contiguous()
    snap = dict(events=ev, times=t, now=t[:, -1].clone(), carry=p.reshape(1, 1, -1).expand(1, B, -1).contiguous(),
                carry_time=t[:, 0] - 0.5)
    return w, snap, F


def test_the_records_reader_switch_reading_on_a_saturated_memory():
    """repair_at_switch on a memory held at +-1 by its candidate's bias: the switch holds, the repaired state is
    not saturated and its candidate has the parts' sizes."""
    from lem.model.persistent_memory import PersistentMemory
    w, snap, F = _saturated_weights()
    out = R.repair_at_switch(w, snap)
    assert out["status"] == "switched"
    assert out["record"]["bias_change_max_abs"] <= out["record"]["bias_change_limit"]
    assert out["state_share_above_0p99"] < 0.5
    # the candidate's root mean square is at most the two sizes added (1 + 0.5; the moments come from the snapshot
    # in two passes), and at a root mean square r at most (r / 2.647)^2 of its values can pass 2.647 (Chebyshev)
    assert out["candidate_rms"] <= 1.6
    assert out["candidate_share_above_tanh_0p99"] <= (out["candidate_rms"] / 2.647) ** 2 + 1e-9
    assert abs(out["candidate_input_scale"] - 1.0) < 1e-3              # the stored rows were rescaled at the switch
    assert set(out) >= {"no_recurrent_cap", "recurrent_cap_equal"}
    assert out["recurrent_cap_equal"]["candidate_recurrent_part_rms"] > out["candidate_recurrent_part_rms"]
    # a memory whose state varies is refused
    torch.manual_seed(2)
    q = PersistentMemory(F, 2, 4, 3, hidden_size=8)
    with torch.no_grad():
        q.readout.weight.normal_(0, 2.0)
    assert R.repair_at_switch({k: v.clone() for k, v in q.state_dict().items()},
                              dict(snap, carry=None, carry_time=None))["status"] == "refused"


def test_the_records_reader_reads_a_switched_memory():
    """The review, finding 8: the reader built a plain memory and the refusal stopped it at every switched
    checkpoint. A switched memory's weights now build the repaired memory, and the reader's state is the raw one
    (what the memory hands out plus its running mean)."""
    from lem.model.persistent_memory import PersistentMemory
    w, snap, F = _saturated_weights(seed=3)
    rep = PersistentMemory(F, 2, 4, 3, hidden_size=8, repair="centred")
    rep.load_state_dict(w, strict=False)
    rep.switch(snap)
    switched = {k: v.clone() for k, v in rep.state_dict().items()}
    pm, n_mod, d = R.build_memory(switched)
    assert pm.repaired and (n_mod, d) == (2, 4)
    raw = R.encode(pm, snap)
    view, _ = pm.encode(**{k: snap[k] for k in ("events", "times", "now", "carry", "carry_time")})
    torch.testing.assert_close(raw, view + pm.state_centre, rtol=0, atol=0)
    plain, _n, _d = R.build_memory(switched, plain=True)
    assert not plain.repaired


def test_the_card_proof_reads_the_fold_bit_for_bit_on_the_cpus_bf16():
    """persistent_memory_card_proof.prove_on on a saturated memory with a constant of hundreds, under the CPU's
    bf16 (the card's own arithmetic is the pause job's): the live bias after the switch equals production's live
    bias at every number; the ring's moments at the reference state give the fold; a ring moment far from the
    reference makes the switch refuse (the review, finding 5)."""
    w, snap, F = _saturated_weights(seed=5, d=64, weight_sd=40.0)
    from lem.model.persistent_memory import PersistentMemory
    plain = PersistentMemory(F, 2, 64, 3, hidden_size=8)
    plain.load_state_dict(w)
    with torch.no_grad():
        state, _ = plain.encode(**snap)
    ring = SimpleNamespace(memory_states=torch.zeros(4, 16, 16), ptr=10, count=10, size=16)
    ring.memory_states[:, :10] = state[:, None, :]
    row = CP.prove_on(dict(step=411150, replay_batch=6, path="C:/x/ckpt_step_411150"), w, snap, ring,
                      torch.device("cpu"), save=lambda r: None)
    assert row["live_bias_dtype"] == "bfloat16" and row["live_after_dtype"] == "float32"
    assert row["live"]["numbers_changed"] == 0 and row["live"]["numbers"] == 4 * 2 * 64
    assert row["ring"]["numbers_above_line"] == 0 and row["ring"]["moments"] == 40
    assert row["switch"]["production_bias_dtype"] == "bfloat16" and row["switch"]["ring_states_above_line"] == 0
    assert row["timing_ms"]["repaired_encode_backward"] > 0
    far = SimpleNamespace(memory_states=ring.memory_states.clone(), ptr=10, count=10, size=16)
    far.memory_states[2, 4] += 0.15
    row = CP.prove_on(dict(step=380550, replay_batch=6), w, snap, far, torch.device("cpu"), save=lambda r: None)
    assert row["switch"]["status"] == "refused" and "written replay moments" in row["switch"]["reason"]


def test_the_vision_tools_read_a_switched_memory(tmp_path):
    """The review, finding 8: vision_memory_ablation.memory_source read the share above 0.99 of what encode
    returns (after the switch the centred state, near 0 whatever the raw state does), and
    vision_regression_records.weights_of read the readout weight's size (0 at the switch: the constant moved into
    the bias) as the memory's. Both now read a switched memory as switched."""
    from safetensors.torch import save_file
    from lem.model.persistent_memory import PersistentMemory
    VA = _module("vision_memory_ablation")
    VR = _module("vision_regression_records")
    w, snap, F = _saturated_weights(seed=7)
    rep = PersistentMemory(F, 2, 4, 3, hidden_size=8, repair="centred")
    rep.load_state_dict(w, strict=False)
    rep.switch(snap)
    out = VA.memory_source(rep, snap)
    view, _ = rep.encode(**{k: snap[k] for k in ("events", "times", "now", "carry", "carry_time")})
    raw = view + rep.state_centre
    assert out["repaired"] is True
    assert out["last_state_share_above_0p99"] == pytest.approx(float((raw[:, :8].abs() > 0.99).float().mean()))
    d = 4
    tensors = {"prediction_heads.heads.vision.attn.in_proj_weight": torch.randn(3 * d, d),
               "prediction_heads.heads.vision.attn.out_proj.weight": torch.randn(d, d),
               "persistent_memory.readout.weight": torch.zeros(6 * d, 16),
               "persistent_memory.readout.bias": torch.full((6 * d,), 3.0),
               "persistent_memory.bias_at_switch": torch.full((6 * d,), 2.0),
               "persistent_memory.repair_switched": torch.tensor(True),
               "forward_model.final_norm.weight": torch.ones(d), "forward_model.final_norm.bias": torch.zeros(d),
               "prediction_heads.heads.vision.mlp.2.weight": torch.randn(d, d),
               "encoder_registry.encoders.vision.w": torch.randn(2, 2)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    r, _dirs, _enc = VR.weights_of(str(tmp_path))
    assert r["memory_switched"] is True and r["memory_readout_norm"] == 0.0
    assert r["memory_constant_rms"] == pytest.approx(3.0) and r["memory_constant_moved_since_switch_rms"] == pytest.approx(1.0)
    tensors["persistent_memory.repair_switched"] = torch.tensor(False)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    assert VR.weights_of(str(tmp_path))[0]["memory_switched"] is False


@pytest.mark.slow
def test_the_lived_step_grades_every_lesson_through_leaves(tmp_path):
    """P2's grading (lived_step_gradients) on a real reach world with the efference and contact heads and the
    memory: every loss term the step builds with a graph is graded on the memory bias leaves; the delayed
    lesson's bias and the lived moment's are both leaves; the next-moment lesson reaches the delayed lesson's
    bias; the needs forecast's lesson is graded on its state; the wrappers are gone afterwards and a plain step
    runs."""
    import numpy as np
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                persistent_memory="on", efference_heads=True, contact_head=True)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "w"), tag="_p2", vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        prev = None
        for i in range(4):
            r = m.step(frames, dev, train=False, step_index=i, executed_action=prev)
            order = r.motor_for_action.copy()
            frames = src.step(order, np.asarray(r.gaze_out)[0:1], None, render=False)
            prev = LemLite.gaze_to_handle(order, np.asarray(r.gaze_out)[0:1])
        with torch.no_grad():
            m.persistent_memory.readout.weight.normal_(0.0, 0.1)   # a readout that passes requests to the state
        out = M.lived_step_gradients(m, lambda: m.step(frames, dev, train=False, step_index=4, executed_action=prev))
        pm = m.persistent_memory
        assert "bias" not in pm.__dict__ and "predict_needs" not in pm.__dict__
        assert "compute" not in m.loss_manager.__dict__
        assert out["state"] is not None and len(out["state"]) == 2 * pm.hidden_size
        assert {"delayed_lesson", "lived_moment"} <= set(out["bias_calls"])
        assert "next_frame" in out["lessons"] and "delayed_lesson" in out["lessons"]["next_frame"]
        assert "needs_forecast" in out["lessons"]
        for name in ("efference_body", "efference_gaze", "contact_head"):
            if name in out["lessons_built"]:
                assert name in out["lessons"], name          # it reads the delayed lesson's latent
        assert set(out["lessons"]) <= set(out["lessons_built"])
        m.step(frames, dev, train=False, step_index=5, executed_action=prev)
    finally:
        src.close()


# ------------------------------------------------------------------ the deploy watcher (persistent_memory_deploy_watch)
def _stats_folder(root, name, rows, mtime):
    import json as _json
    import os as _os
    d = root / name
    d.mkdir()
    p = d / "replay_training_stats.jsonl"
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _os.utime(p, (mtime, mtime))
    return d


def _mimo(step, ratios, rate=0.5, spread=0.4, goal=None, gate=0.0):
    """A full report: every graded sense's audit with 800 transitions and the given ratio (persistence 1); goal: the
    memory block's goal readings without body 2 for every graded sense (help, information, control), or None."""
    senses = ("vision", "tactile", "audio", "proprioception")
    g = {}
    if goal is not None:
        h, i, c = goal
        g = {s: dict(without_body_2=dict(ln_without_over_with=h, ln_other_over_with=i, ln_other2_over_other=c),
                     all_bodies=dict(ln_without_over_with=h, ln_other_over_with=i, ln_other2_over_other=c))
             for s in senses}
    return dict(type="mimo", step=step, model_steps_per_s=rate,
                prediction_audit=dict(modalities={s: dict(transitions=800, prediction_mse=ratios[s],
                                                          persistence_mse=1.0) for s in senses}),
                prediction_audit_per_body=dict(senses={s: dict(without_body_2=dict(ratio=ratios[s]))
                                                       for s in senses}),
                persistent_memory=dict(state_spread_between_bodies=spread, state_spread_over_window=spread,
                                       update_gate_share_above_0p99=gate, candidate_input_scale=1.0,
                                       candidate_recurrent_scale=1.0, candidate_input_rows_norm=7.2,
                                       candidate_recurrent_rows_norm=1.67, constant_bias_moved_rms=0.01,
                                       **({"goal": g} if g else {})))


def test_the_deploy_watcher_reads_across_segments_and_judges_the_declared_lines(tmp_path):
    """The watcher reads the 20 full reports before the switch from an earlier segment folder, finds the switch
    record, and judges K2, K4, K5 and the goal; a run that changes nothing stops on nothing; a K4, K2 or K5
    crossing stops, each under one key per criterion and switch."""
    W = _module("persistent_memory_deploy_watch")
    base = dict(vision=2.0, tactile=1.03, audio=0.9, proprioception=1.01)
    lines = dict(K1={s: 1.0 for s in base}, K4=dict(vision=0.944, tactile=0.0282, audio=0.052, proprioception=0.0138),
                 K5=0.862)
    sw = dict(type="persistent_memory_switch", step=3000, part="switch",
              switch=dict(switched_at_step=3000, pre_switch_checkpoint="D:/x/ckpt_step_3000", bias_change_max_abs=1e-5,
                          production_change_max_abs=0.0, bias_change_limit=2e-4, ring_states_above_line=0,
                          candidate_input_rows_norm_after_switch=7.2, candidate_recurrent_rows_norm_after_switch=1.67))

    def helps(k):                     # the own memory helps and informs; the control alternates in sign
        return (0.02, 0.015, 0.004 * (1 if k % 2 else -1))

    def lineage(after_ratios, spread=0.4, goal=helps, rate=0.5):
        root = tmp_path / ("lin_%d" % len(list(tmp_path.iterdir())))
        root.mkdir()
        _stats_folder(root, "seg1", [_mimo(100 * k, base) for k in range(1, 31)], 1000.0)
        _stats_folder(root, "seg2", [sw] + [_mimo(3000 + 100 * k, after_ratios, rate=rate, spread=spread,
                                                  goal=goal(k)) for k in range(0, 121)], 2000.0)
        return W.read_lineage(str(root / "seg*"))

    mimo, switches = lineage(base)
    v = W.judge(mimo, switches, lines)
    assert v["state"] == "switched" and v["switch_step"] == 3000 and not v["stop"], v["stops"]
    assert v["readings"]["reports"]["before"] == 20 and v["readings"]["K0"]["held"]
    assert abs(v["readings"]["K4"]["ln_over_median_before"]["vision"]) < 1e-12
    g = v["readings"]["GOAL"]
    assert g["met"] and g["met_by"] == ["vision", "tactile", "audio", "proprioception"]
    assert g["senses"]["vision"]["margin"] == pytest.approx(0.004) and g["senses"]["vision"]["control_above_0"] == 10
    worse = dict(base, proprioception=1.01 * math.exp(0.02))
    v = W.judge(*lineage(worse, goal=lambda k: (0.0, 0.0, 0.004 * (1 if k % 2 else -1))), lines)
    assert v["stop"] and any(x.startswith("K4 proprioception") for x in v["stops"])
    assert v["stop_keys"] == ["K4 proprioception|seg2@3000"]
    assert not v["readings"]["GOAL"]["met"] and v["readings"]["GOAL"]["note"].startswith("MISSED")
    v = W.judge(*lineage(base, spread=0.05), lines)
    assert v["stop"] and any(x.startswith("K2 between") for x in v["stops"])
    assert v["stop_keys"] == ["K2 between|seg2@3000", "K2 window|seg2@3000"]      # one key per criterion, not per report
    v = W.judge(*lineage(base, rate=0.4), lines)
    assert v["stop"] and any(x.startswith("K5") for x in v["stops"])
    assert not any("raises" in x for x in v["alerts"])
    # the alert beside K4: the varying part raising every sense's error on reports 2-21 (no stop)
    v = W.judge(*lineage(base, goal=lambda k: (-0.02, -0.02, 0.004 * (1 if k % 2 else -1))), lines)
    assert not v["stop"] and sum("raises" in x for x in v["alerts"]) == 4
    assert v["readings"]["K4"]["memory_harm"]["vision"]["help_below_0"] == 20


def test_the_goal_rule_needs_help_and_information_beyond_the_paired_noise():
    """The redeclared goal (the third review's finding 4) on one sense's 20 reports: help and information each above 0
    in 16 or more and each median above the margin (the median |control|) meets it; a memory whose presence helps
    but whose own variation beats another body's no more than chance (information about 0) misses it; a variation
    smaller than the paired noise misses it; a one-sided control gives no verdict; fewer than 20 reports, none."""
    W = _module("persistent_memory_deploy_watch")
    alt = [0.01 * (1 if k % 2 else -1) for k in range(20)]

    def blocks(help_, info, ctrl):
        return [dict(ln_without_over_with=h, ln_other_over_with=i, ln_other2_over_other=c)
                for h, i, c in zip(help_, info, ctrl)]
    assert W.goal_sense(blocks([0.05] * 20, [0.03] * 20, alt))["verdict"] == "met"
    info_chance = [0.03 * (1 if k % 2 else -1) for k in range(20)]
    g = W.goal_sense(blocks([0.05] * 20, info_chance, alt))
    assert g["verdict"] == "missed" and g["information_above_0"] == 10
    g = W.goal_sense(blocks([0.005] * 20, [0.005] * 20, alt))
    assert g["verdict"] == "missed" and g["margin"] == pytest.approx(0.01)       # 0.005 is inside the paired noise
    g = W.goal_sense(blocks([0.05] * 20, [0.03] * 20, [0.01] * 20))
    assert g["verdict"].startswith("no verdict: the control is above 0 in 20")
    assert W.goal_sense(blocks([0.05] * 19, [0.03] * 19, alt[:19]))["verdict"].startswith("no verdict: fewer")


# ------------------------------------------------ the watcher's --act on planted lineages (the third review, finding 3)
_SW = dict(type="persistent_memory_switch", step=3000, part="switch",
           switch=dict(switched_at_step=3000, pre_switch_checkpoint="seg1/checkpoints/ckpt_step_3000",
                       bias_change_max_abs=1e-5, production_change_max_abs=0.0, bias_change_limit=2e-4,
                       ring_states_above_line=0, candidate_input_rows_norm_after_switch=7.2,
                       candidate_recurrent_rows_norm_after_switch=1.67))
_BASE = dict(vision=2.0, tactile=1.03, audio=0.9, proprioception=1.01)


def _run(exp, name, rows, mtime, resume=None):
    """A run folder as the trainer and the supervisor leave it: its stats stream and args.json (the checkpoint it
    resumed, whose folder's folder names the parent run)."""
    import json as _json
    d = _stats_folder(exp, name, rows, mtime)
    (d / "args.json").write_text(_json.dumps({"resume": resume}), encoding="utf-8")
    return d


def _watch_setup(tmp_path, monkeypatch, W, active, paused=False, alive=True):
    """The watcher's view of the machine: training_continuation.json naming the active run, supervisor_control.json,
    the watcher's own record of what it acted on, whether a trainer is alive; and a lines file."""
    import json as _json
    cont, ctl, state = tmp_path / "continuation.json", tmp_path / "supervisor_control.json", tmp_path / "state.json"
    cont.write_text(_json.dumps({"active_run": str(active), "trainer_pids": []}), encoding="utf-8")
    ctl.write_text(_json.dumps({"cmd": "pause" if paused else "resume"}), encoding="utf-8")
    monkeypatch.setattr(W, "CONTINUATION", str(cont), raising=False)
    monkeypatch.setattr(W, "SUPERVISOR_CONTROL", str(ctl), raising=False)
    monkeypatch.setattr(W, "STATE", str(state), raising=False)
    monkeypatch.setattr(W, "trainer_alive", lambda *a, **k: alive, raising=False)
    lines = tmp_path / "lines.json"
    if not lines.exists():
        lines.write_text(_json.dumps({"lines": dict(
            **{"K1_" + s: {"line": 1.0} for s in _BASE},
            K4_vision={"line": 0.944}, K4_tactile={"line": 0.0282}, K4_audio={"line": 0.052},
            K4_proprioception={"line": 0.0138}, K5={"line": 0.862})}), encoding="utf-8")
    return lines


def _watch(W, tmp_path, lines, act_folder, exp):
    """One watcher run, --act given the folder the deploy procedure named (the old interface)."""
    import json as _json
    out = tmp_path / "verdict.json"
    code = W.main(["--act", str(act_folder), "--lineage", str(exp / "seg*"), "--lines", str(lines),
                   "--out", str(out), "--from-step", "0"])
    return code, _json.loads(out.read_text(encoding="utf-8"))


def _switched_lineage(tmp_path, spread=0.05):
    """seg1: 30 full reports before the switch; seg2: the switching segment (resumed from seg1's 3,000), its memory's
    spread between bodies at `spread` from the switch on (0.05 crosses K2's 0.08)."""
    exp = tmp_path / "exp"
    exp.mkdir()
    _run(exp, "seg1", [_mimo(100 * k, _BASE) for k in range(1, 31)], 1000.0)
    s = _run(exp, "seg2", [_SW] + [_mimo(3000 + 100 * k, _BASE, spread=spread) for k in range(0, 21)], 2000.0,
             resume=str(exp / "seg1" / "checkpoints" / "ckpt_step_3000"))
    return exp, s


def test_the_watcher_writes_no_stop_in_a_pause_with_no_trainer(tmp_path, monkeypatch):
    """P1: a crossed line in a coordinated pause with no trainer alive writes nothing (production's H84 watcher fix,
    4f8d34ae7); the same crossing with a live trainer writes the stop."""
    W = _module("persistent_memory_deploy_watch")
    exp, s = _switched_lineage(tmp_path)
    lines = _watch_setup(tmp_path, monkeypatch, W, s, paused=True, alive=False)
    code, v = _watch(W, tmp_path, lines, s, exp)
    assert code == 2 and v["stop"]
    assert not (s / "control.json").exists(), v.get("acted")
    lines = _watch_setup(tmp_path, monkeypatch, W, s, paused=True, alive=True)
    code, v = _watch(W, tmp_path, lines, s, exp)
    assert (s / "control.json").read_bytes() == b'{"cmd": "snapshot_and_stop"}'


def test_the_watcher_never_writes_a_stop_again_after_it_was_carried_out(tmp_path, monkeypatch):
    """P2: the trainer consumed the stop (control_done_*.json and an overseer_cmd record); the next run writes
    nothing for the same crossing."""
    import json as _json
    import time as _time
    W = _module("persistent_memory_deploy_watch")
    exp, s = _switched_lineage(tmp_path)
    lines = _watch_setup(tmp_path, monkeypatch, W, s)
    _watch(W, tmp_path, lines, s, exp)
    assert (s / "control.json").exists()
    now = int(_time.time())
    (s / "control.json").replace(s / ("control_done_%d.json" % now))
    with open(s / "replay_training_stats.jsonl", "a", encoding="utf-8") as f:
        f.write(_json.dumps({"type": "overseer_cmd", "timestamp": now + 0.5, "step": 5000,
                             "cmd": "snapshot_and_stop"}) + "\n")
    code, v = _watch(W, tmp_path, lines, s, exp)
    assert code == 2 and v["stop"]                        # the crossing is still reported
    assert not (s / "control.json").exists(), v.get("acted")


def test_the_watcher_delivers_the_stop_to_the_live_run_after_a_relaunch(tmp_path, monkeypatch):
    """P3: the stop written into the switching segment was never consumed (the trainer ended first) and the
    supervisor relaunched the switched run into a new folder; the watcher, still given the old folder, must deliver
    the stop to the live run named by training_continuation.json."""
    W = _module("persistent_memory_deploy_watch")
    exp, s = _switched_lineage(tmp_path)
    lines = _watch_setup(tmp_path, monkeypatch, W, s)
    _watch(W, tmp_path, lines, s, exp)
    assert (s / "control.json").exists()
    loaded = dict(_SW, part="loaded", step=5000)
    r = _run(exp, "seg2_resume_1", [loaded] + [_mimo(5000 + 100 * k, _BASE, spread=0.05) for k in range(1, 5)],
             3000.0, resume=str(exp / "seg2" / "checkpoints" / "ckpt_step_5000"))
    lines = _watch_setup(tmp_path, monkeypatch, W, r)
    code, v = _watch(W, tmp_path, lines, s, exp)
    assert code == 2 and (r / "control.json").read_bytes() == b'{"cmd": "snapshot_and_stop"}', v.get("acted")


@pytest.mark.parametrize("vision_after", [1.0, 2.7])
def test_the_watcher_ends_its_watch_after_the_auto_revert(tmp_path, monkeypatch, vision_after):
    """P4 and P4b: the auto-revert resumes the pre-switch checkpoint without the flag in a new folder. That run does
    not descend from the switching folder: the watcher judges no switch, reads no goal on it and never stops it,
    even when its vision reads 2.7 times worse on reports 2-21 (which crosses K4 on a switched run)."""
    W = _module("persistent_memory_deploy_watch")
    exp, s = _switched_lineage(tmp_path, spread=0.4)
    worse = dict(_BASE, vision=_BASE["vision"] * vision_after)
    rows = [_mimo(3000 + 100 * k, worse if 1 <= k <= 21 else _BASE) for k in range(0, 121)]
    r = _run(exp, "seg_revert", rows, 3000.0, resume=str(exp / "seg1" / "checkpoints" / "ckpt_step_3000"))
    lines = _watch_setup(tmp_path, monkeypatch, W, r)
    code, v = _watch(W, tmp_path, lines, r, exp)
    assert v["state"] != "switched" and not v["stop"] and code == 0, v["state"]
    assert "GOAL" not in v.get("readings", {})
    assert not (r / "control.json").exists()


def test_the_switch_trial_builds_three_arms_from_productions_command_and_compares_them():
    """persistent_memory_switch_trial.py: the supervisor's command with the lineage's folders, the schedules and the
    resume replaced, the flag added only in the "on" arm and never kept from the command, and never a checkpoint
    interval of 0 (the third review's finding 1); two flag-off arms around the "on" arm (finding 5); the comparison
    reads each report's ratio pooled and without body 2, the drift between the flag-off arms, and pools per 100
    updates for the declared reading on K1's window."""
    T = _module("persistent_memory_switch_trial")
    base = ["py", "-m", "mimo_bridge.train_mimo", "--out", "D:/runs/seg41", "--resume", "D:/runs/seg40/ckpt_step_9",
            "--persistent-memory", "on", "--persistent-memory-repair", "off", "--report-every", "100",
            "--checkpoint-every-min", "60", "--minutes", "480", "--n-envs", "8",
            "--persistent-memory-constant-return", "7"]
    assert [a[0] for a in T.ARMS] == ["off", "on", "on_return", "off2"]
    assert [a[1] for a in T.ARMS] == [False, True, True, False] and [a[2] for a in T.ARMS] == [None, 0, 2000, None]
    on = T.arm_command(base, "D:/ck/ckpt_step_415025", Path("D:/tmp/on"), 200, 20, True)
    off = T.arm_command(base, "D:/ck/ckpt_step_415025", Path("D:/tmp/off"), 200, 20, False)
    for cmd in (on, off):
        assert "D:/runs/seg41" not in cmd and "D:/runs/seg40/ckpt_step_9" not in cmd and "480" not in cmd
        assert cmd[cmd.index("--resume") + 1] == "D:/ck/ckpt_step_415025"
        assert cmd[cmd.index("--max-steps") + 1] == str(415025 + 200)
        assert cmd[cmd.index("--report-every") + 1] == "20" and cmd.count("--report-every") == 1
        assert cmd[cmd.index("--n-envs") + 1] == "8" and cmd[cmd.index("--persistent-memory") + 1] == "on"
        assert cmd.count("--checkpoint-every-min") == 1 and float(cmd[cmd.index("--checkpoint-every-min") + 1]) >= 1e4
        assert cmd[cmd.index("--minutes") + 1] == "0"            # --max-steps ends the run
    assert on[on.index("--persistent-memory-repair") + 1] == "centred" and "--persistent-memory-repair" not in off
    assert on[on.index("--persistent-memory-constant-return") + 1] == "0" and on.count("--persistent-memory-constant-return") == 1
    assert "--persistent-memory-constant-return" not in off
    ret = T.arm_command(base, "D:/ck/ckpt_step_415025", Path("D:/tmp/r"), 200, 20, True, 2000)
    assert ret[ret.index("--persistent-memory-constant-return") + 1] == "2000"

    def rec(step, v, wo, rate=0.5):
        return dict(type="mimo", step=step, model_steps_per_s=rate,
                    prediction_audit=dict(modalities=dict(vision=dict(prediction_mse=v, persistence_mse=1.0))),
                    prediction_audit_per_body=dict(senses=dict(vision=dict(without_body_2=dict(ratio=wo)))))
    c = T.compare([rec(20, 2.0, 1.0), dict(type="persistent_memory_switch", part="switch", switch={})],
                  [rec(20, 1.0, 2.0)], [rec(20, 1.5, 2.0)])
    row = c["rows"][0]
    assert row["ln_ratio_on_over_off"]["vision"] == pytest.approx(math.log(2.0))
    assert row["ln_ratio_on_over_off_without_body_2"]["vision"] == pytest.approx(math.log(0.5))
    assert row["ln_ratio_off2_over_off"]["vision"] == pytest.approx(math.log(1.5))
    assert len(c["switch"]) == 1
    # pooled per 100 updates from the resumed checkpoint: summed errors over summed "stays the same" errors
    on_s = [rec(1000 + 20 * k, 1.0 + 0.1 * k, 1.0) for k in range(1, 11)]
    off_s = [rec(1000 + 20 * k, 1.0, 1.0) for k in range(1, 11)]
    c = T.compare(on_s, off_s, off_s, start=1000)
    assert [p["updates"] for p in c["pooled"]] == [[1001, 1100], [1101, 1200]]
    assert c["pooled"][0]["ln_ratio_on_over_off"]["vision"] == pytest.approx(math.log((5 + 0.1 * 15) / 5))
    assert c["pooled"][0]["ln_ratio_off2_over_off"]["vision"] == pytest.approx(0.0)
    d = c["declared_first_window"]["vision"]
    assert d["k1_line"] == 1.40 and d["verdict"] == "inside"
    worse = [rec(1000 + 20 * k, 2.0, 1.0) for k in range(1, 11)]
    c = T.compare(off_s, off_s, off_s, start=1000, on_return=worse)
    r = c["declared_return"][0]["senses"]["vision"]
    assert r["ln_on_return_over_on"] == pytest.approx(math.log(2.0)) and r["verdict"] == "inside"   # vision's line 0.944
    assert r["frozen_heads_at_a_tenth"] == pytest.approx(8.7042)
    drift = [rec(1000 + 20 * k, 5.0, 1.0) for k in range(1, 11)]
    assert T.compare(on_s, off_s, drift, start=1000)["declared_first_window"]["vision"]["verdict"].startswith(
        "no verdict")


@pytest.mark.slow
def test_the_switch_trials_command_writes_at_most_one_checkpoint(tmp_path, monkeypatch):
    """The third review's finding 1, through the real trainer on the CPU (one body): a small run writes a checkpoint;
    the trial's arm command built from a command with an hourly checkpoint resumes it for 4 updates and writes at
    most one checkpoint; the control, the same arm with the interval at 0, writes one per update."""
    import json as _json
    from mimo_bridge.tools import preflight
    from mimo_bridge.train_mimo import main
    monkeypatch.setattr(preflight, 'run_probes', lambda path, env_kw=None: dict(ok=True, checks=[]))
    torch.set_num_threads(1)
    T = _module("persistent_memory_switch_trial")
    flags = ['--device', 'cpu', '--n-envs', '1', '--no-vision', '--curriculum', 'off', '--age-binding', 'none',
             '--delay-binding', 'off', '--episode-steps', '0', '--action-repeat', '1', '--motor-hold', '1',
             '--no-babble', '--body-function', 'modulate', '--persistent-memory', 'on', '--imagination', 'off',
             '--next-frame-loss', 'on', '--contact-profile', 'mechanical_v4', '--replay', 'off',
             '--decoder-kp', '6.0']
    first = tmp_path / "first"
    assert main(flags + ['--out', str(first), '--minutes', '0', '--max-steps', '2', '--report-every', '1',
                         '--video-every-min', '0', '--checkpoint-every-min', '60',
                         '--checkpoint-storage', str(first / 'checkpoints')]) == 0
    ck = Path(_json.loads((first / 'summary.json').read_text())["checkpoints"][-1]["path"])
    base = ["py", "-m", "mimo_bridge.train_mimo"] + flags + [
        '--out', 'X', '--checkpoint-storage', 'Y', '--minutes', '480', '--max-steps', '0', '--report-every', '100',
        '--checkpoint-every-min', '60', '--video-every-min', '20']
    arm = tmp_path / "arm"
    cmd = T.arm_command(base, str(ck), arm, 4, 2, False)
    assert main(cmd[3:]) == 0
    written = sorted(p.name for p in (arm / "checkpoints").glob("ckpt_step_*"))
    assert len(written) <= 1, written
    # the control: the same arm with the interval at 0 saves at every update, so this test can fail
    monkeypatch.setattr(T, "CHECKPOINT_EVERY_MIN", 0)
    zero = tmp_path / "zero"
    assert main(T.arm_command(base, str(ck), zero, 4, 2, False)[3:]) == 0
    assert len(list((zero / "checkpoints").glob("ckpt_step_*"))) >= 4


def test_the_goal_reach_reading_pairs_each_bodys_bias_with_the_mean_and_other_bodies():
    """persistent_memory_goal_reach.py (the record's NEXT 6): the arms (own, the bodies' mean, body b + 1's, body
    b + 2's), the blocks spread over the ring, and the readings' arithmetic: a sense whose own-bias error is lower
    than every other arm's in every block meets the goal rule; body 2 is left out of the without-body-2 pooling."""
    G = _module("persistent_memory_goal_reach")
    W = _module("persistent_memory_deploy_watch")
    bias = torch.arange(3.).reshape(3, 1, 1).expand(3, 2, 2)
    arms = G.arm_biases(bias)
    assert torch.equal(arms["other"][0], bias[1]) and torch.equal(arms["other2"][0], bias[2])
    assert torch.equal(arms["other"][2], bias[0]) and float(arms["mean"][1, 0, 0]) == 1.0
    backs = G.positions(6000, 129, 20, 8, 12)
    assert len(backs) == 20 and all(len(b) == 8 for b in backs)
    assert backs[0][0] == 0 and max(max(b) for b in backs) <= 6000 - 129
    B = 4

    def block(k):
        sign = 1.0 if k % 2 else -1.0
        e = {"own": [1.0] * B, "mean": [1.2] * B, "other": [1.1] * B, "other2": [1.1 * (1 + 0.01 * sign)] * B}
        e["own"][2] = 5.0                                      # body 2 is left out of the without-body-2 pooling
        return {"tactile": {"err": e, "pers": [2.0] * B, "n": [8] * B}}
    s = G.summarize([block(k) for k in range(20)], W.goal_sense)["tactile"]
    wo = s["without_body_2"]
    assert wo["pooled"]["ln_help"] == pytest.approx(math.log(1.2))
    assert wo["pooled"]["ln_information"] == pytest.approx(math.log(1.1))
    assert wo["pooled"]["help_in_100"] == pytest.approx(100 * (1 - 1 / 1.2))
    assert wo["goal_rule"]["verdict"] == "met" and wo["goal_rule"]["control_above_0"] == 10
    assert s["all_bodies"]["pooled"]["ln_help"] < 0                  # body 2's own error of 5 dominates the pooling


def test_the_bias_size_acceptance_reads_each_slot_against_its_pre_deploy_size():
    """The coordinator's acceptance (2026-09-26): each sense slot's memory bias back near its pre-deploy size and not
    regrowing. Over the first 20 full reports from the return's length on, every slot at or under its line (the
    344,400 size plus its gap to 247,000) is met; the share of the gap closed reads 0% at the switch and about 100%
    at the end; a slot that was under its target at the switch has no gap; a slot whose varying part grows past its
    line after the return is an alert."""
    W = _module("persistent_memory_deploy_watch")
    slots = list(W.PRE_DEPLOY_BIAS)
    start = dict(vision=105.5, tactile=267.5, audio=5.63, proprioception=207.5, motor=128.1, vestibular=6.67)
    swr = dict(constant_return=dict(updates=1000, start_per_slot=start))

    def rep(step, varying):
        p = min(1.0, (step - 3000) / 1000.0)
        eff = {s: (start[s] * (W.PRE_DEPLOY_BIAS[s] / start[s]) ** p if start[s] > W.PRE_DEPLOY_BIAS[s] else start[s])
               for s in slots}
        return dict(step=step, persistent_memory=dict(constant_effective_rms_per_slot=eff,
                                                      varying_bias_rms_per_slot={s: varying for s in slots}))
    after = [rep(3000 + 100 * k, 0.5) for k in range(0, 31)]
    out = W.bias_size(after, 3000, swr)
    assert out["met"] and out["judged_at"] == 3000 + 100 * 29 and not out["regrown"]
    want = 100 * (267.5 - math.sqrt(6.11 ** 2 + 0.25)) / (267.5 - 6.11)
    assert out["gap_closed_percent"]["tactile"] == pytest.approx(want)
    assert out["gap_closed_percent"]["audio"] is None
    assert W.bias_size(after[:5], 3000, swr)["note"].startswith("not yet")
    out = W.bias_size(after + [rep(6200, 9.0)], 3000, swr)
    assert set(out["regrown"]) == set(slots)
    assert W.bias_size(after, 3000, {})["note"] == "no constant return in the switch record"


def test_the_constant_return_reading_moves_each_slot_and_sets_the_schedule_by_the_declared_rule():
    """persistent_memory_constant_return.py: a slot's constant moved a fraction of the way (geometrically) to its
    target, what differs between the bodies kept, a slot under its target left; the schedule's length is the
    smallest whose fractions at updates 100 and 2,100 stay inside the K1 and K4 lines."""
    CR = _module("persistent_memory_constant_return")
    bias = torch.zeros(3, 2, 4)
    bias[:, 0] = 100.0
    bias[:, 1] = 2.0
    bias[0, 0, 0] += 1.0
    moved = CR.moved_constant(bias, torch.tensor([10.0, 5.0]), 0.5)
    c = moved.mean(0)
    r0 = float(bias.mean(0)[0].square().mean().sqrt())
    assert float(c[0].square().mean().sqrt()) == pytest.approx(math.sqrt(r0 * 10.0), rel=1e-5)
    assert torch.allclose(moved[:, 1], bias[:, 1])                               # under its target: left
    assert torch.allclose(moved - c, bias - bias.mean(0, keepdim=True), atol=1e-4)  # the bodies' differences kept
    lines = dict(vision=1.0, tactile=0.1, audio=0.1, proprioception=0.1)
    readings = {0.0: dict.fromkeys(lines, 0.0), 0.1: dict(vision=0.5, tactile=0.05, audio=0.0, proprioception=0.0),
                0.5: dict(vision=2.0, tactile=0.5, audio=0.0, proprioception=0.0), 1.0: dict.fromkeys(lines, 3.0)}
    got = CR.schedule_length(readings, k4=lines, k1=lines)
    assert got["updates"] == 21000                                               # 2,100 / 0.1
    assert CR.schedule_length({0.0: dict.fromkeys(lines, 0.0), 0.1: dict.fromkeys(lines, 9.0)},
                              k4=lines, k1=lines)["updates"] is None


def test_the_ancestry_follows_the_recorded_parent_of_a_run_resumed_from_a_copied_checkpoint(tmp_path):
    """H84 segment 41 (2026-09-29): the smoke resumed a verified D: copy of 415,075, whose folder holds no stats;
    its continuation_receipt.json names the run the checkpoint came from, and the walk must continue there."""
    import json
    W = _module("persistent_memory_deploy_watch")
    exp, copy = tmp_path / "experiments", tmp_path / "copy" / "start"
    parent, child = exp / "H84_seg40", exp / "H84_seg41_smoke"
    for folder, resume in ((parent, None), (child, copy / "checkpoints" / "ckpt_step_415075")):
        folder.mkdir(parents=True)
        (folder / "args.json").write_text(json.dumps({"resume": None if resume is None else str(resume)}))
    (child / "continuation_receipt.json").write_text(json.dumps({"parent_run": str(parent)}))
    assert W.ancestry(str(child)) == [str(parent), str(child)]
    # without the receipt the walk looks for a sibling named after the copy's folder and stops at the child
    (child / "continuation_receipt.json").unlink()
    assert W.ancestry(str(child)) == [str(child)]
