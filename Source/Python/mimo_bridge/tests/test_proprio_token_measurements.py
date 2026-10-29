# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/proprio_token_measurements.py (sixth version) and the pace job
(proprio_token_pace_check.py): the pause measurement's pieces on a lite model and a stand-in ring (no checkpoint of
production is read here), and the version-6 rules on H84's committed readings.

What is pinned: VICReg's one-sense share equals the loss manager's own computation; the next-moment lesson built from a
ring window grades every sense, reaches the proprioception encoder and nothing else, and its proprioception term splits
exactly into the residual path at weight 1 and the core path at the sense's weight; the Adam moments map onto the
encoder by name; the whole-ring decoder is the review's own; the calibration reads the checkpoint's own encoder over the
whole window with the card's arithmetic, its ratio block is the measure job's reading of the same window; the rules
(LINE the pre-deploy median, DEAF the tolerance bound kept out of sample, the log noise robust to a body that changed
state, K6's fall, K2's line, the content control's noise, SKIP, the pre-deploy state) on a
synthetic population and on H84's committed readings; version 4's rows serve them; the switch test loads the flagged
build with a line; the pace job's first-order pace agrees with a finite step, its path reaches the line and stops
pushing there, and a ring extract serves only its own checkpoint; every re-fit arm starts from the same encoder and
puts it back; the clip factor is read from the run's training records; one production update's effect agrees with a
finite step; the grading probe accounts for the whole step; rule LS on the ratio (targets under DEAF, the posture-only
lift, the content control, the skipped bodies) and NF; rule P; a stop mid-checkpoint keeps every stage. Version 6: the
calibration reads the posture's own jitter share, the fatigue answer and the heading reading; its rules draw K2 on the
token's jitter share minus the posture's (post-deploy pairs in its noise) and the fatigue answer's noise, and a
version-5 result is refused; rule LS judges the gain-matched content (a lift without posture information fails) and the
fatigue answer against the live bodies' largest at update 0; arm R needs DEAF; the switch test's flagged build carries
DEAF; the pace job's hinge engages under DEAF, its hold reads the lesson's formula and acts only on pushed bodies, and
its held-out reading reads the second half through the moved encoder with the heading reading. Version 7 (record section
47 and its fix round after the review proprio_token_v7_review_2026-09-27.md): the pace job's A1 line is version 6's
settled line (DEAF-based) when DEAF is given, and its hold's line is the lesson's (the live bodies' median answer on the
posture-only rows); rule LS on paired re-fits in two stages (stage 1 on seeds 13-20, stage 2 on seeds 21-28 only where
look 1 left a target unresolved; each look at Student's one-sided 97.5-in-100 bound, checked against scipy's quantiles;
lifted / not lifted / unresolved, final after look 2; a lost body has a size, a loss the pairs cannot bound is printed;
the content and fatigue lines on the pairs' means; the horizon at the mean pace, a STRENGTH raise a new version; every
declared pair needed; every pair's gain printed beside the verdict; the hold's share and the attribution's bound
reported) and NF on the pairs' paired bound at each look; measure runs its pairs on one set of clip factors, each pair's
arms on one seed, and an arm on those factors draws its seed's moments; arm R's rows carry the hold's gauges; stage 2
runs only where stage 1 left a checkpoint unresolved; the result file is written whole as each pair ends and a rerun
keeps the earlier file; the clip factor walks the parent runs for its 20 training records. Fix round 2 (the re-review
proprio_token_v7_rereview_2026-09-27.md): NF reads a sense no pair can read (the dead balance sense's None) as not read,
never judged, and judges a partly read sense only on at least 8 readable pairs; stage 2 runs where only such a sense
is unread and never where NF missed; a write cut off half way leaves the last whole result file; arm R counts the
hold's updates once a live body gives its line; the pace job's hold line is judged on the posture-only answers."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
MEAS = REPO / "docs" / "gain_of_function_v4" / "proprio_token_measurements.py"
H84_ROWS = REPO / "docs" / "gain_of_function_v3" / "measurements" / "proprio_token_calibrate_v4_2026-09-26.json"


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("proprio_token_measurements", MEAS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vicregs_one_sense_share_is_the_loss_managers_own(M):
    from lem.training.loss_manager import LossManager
    lm = LossManager.__new__(LossManager)
    lm._vicreg_enabled, lm._vicreg_temporal_window = True, 0
    lm._vicreg_variance_target, lm._vicreg_variance_weight, lm._vicreg_covariance_weight = 1.0, 1.0, 0.04
    lm._vicreg_warned_batch, lm._vicreg_history = False, {}
    z = torch.randn(8, 32)
    full = LossManager.compute_vicreg(lm, {"proprioception": z.unsqueeze(1)})
    assert float(M.vicreg_term(z, 1)) == pytest.approx(float(full), rel=1e-6)
    assert float(M.vicreg_term(z, 6)) == pytest.approx(float(full) / 6.0, rel=1e-6)


def _lite(tmp_path=None):
    torch.set_num_threads(1)
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1", overrides={"model": {"residual_prediction": True}},
                   next_frame_norm="persistence", persistent_memory="on",
                   checkpoint_dir=None if tmp_path is None else str(tmp_path))


def _stand_in_ring(model, B=2, N=2400, seed=0):
    """A replay ring in the companion's layout: body 0 a posture that wanders, body 1 the same plus a constant."""
    g = torch.Generator().manual_seed(seed)
    M_ = len(model.active)
    d = model.d_model
    H = len(model.handle_names)
    walk = torch.cumsum(0.004 * torch.randn(B, N, H, 10, generator=g), dim=1)
    ach = torch.zeros(B, N, H, 10)
    ach[..., 0:3] = 0.2 * torch.randn(B, 1, H, 3, generator=g) + walk[..., 0:3]
    ach[..., 3:7] = torch.nn.functional.normalize(torch.randn(B, 1, H, 4, generator=g) + walk[..., 3:7], dim=-1)
    ach[..., 9] = 1.0
    tokens = torch.nn.functional.layer_norm(torch.randn(B, N, M_, d, generator=g), (d,))
    valid = torch.ones(B, N, M_, dtype=torch.bool)
    for k, m in enumerate(model.active):
        if m == "intermind":
            valid[:, :, k] = False
    hidden = 2 * model.persistent_memory.hidden_size
    rep = dict(mods=list(model.active), size=N, ptr=0, count=N, n_envs=B,
               tokens=tokens.to(torch.float16), achieved=ach.to(torch.float16), valid=valid,
               executed=(0.01 * torch.randn(B, N, H, 9, generator=g)).to(torch.float16),
               has_executed=torch.ones(B, N, dtype=torch.bool), episode=torch.zeros(B, N, dtype=torch.int32),
               memory_states=torch.tanh(torch.randn(B, N, hidden, generator=g)))
    nf_raw = torch.zeros(B, H * 12)
    nf_raw.view(B, H, 12)[..., 10] = 0.8
    return rep, nf_raw


def _prime_norm(model):
    """Running persistence errors that make the weights differ from 1 (proprioception quiet: weight above 1)."""
    model.nf_norm.update({"tactile": 0.02, "audio": 0.05, "proprioception": 0.002})


def test_the_window_lesson_grades_every_sense_and_reaches_only_the_proprioception_encoder(M):
    m = _lite()
    _prime_norm(m)
    rep, nf_raw = _stand_in_ring(m)
    ring = M.Ring(rep, nf_raw)
    assert ring.x.shape == (2, 2400, 312)
    enc, encode = M.encoder_of(m)
    for p in m.params:
        p.requires_grad_(False)
    for p in enc.parameters():
        p.requires_grad_(True)
    o = M.next_moment_terms(m, ring, 1500, encode, split=True)
    assert set(o["terms"]) == {"tactile", "audio", "proprioception"}, "intermind is not real: never graded"
    assert o["weights"]["proprioception"] > 1.0 > o["weights"]["audio"]
    total = sum(o["terms"].values())
    assert math.isfinite(float(total))
    grads = torch.autograd.grad(total, list(enc.parameters()), retain_graph=True)
    assert all(float(gg.abs().sum()) > 0 for gg in grads)
    params = list(enc.parameters())
    gw = torch.autograd.grad(o["terms"]["proprioception"], params, retain_graph=True)
    gr = torch.autograd.grad(o["residual_path"], params, retain_graph=True)
    gc_ = torch.autograd.grad(o["core_path"], params, retain_graph=True)
    for a, b, c in zip(gw, gr, gc_):
        assert torch.allclose(a, b + c, atol=1e-7, rtol=1e-4)
    assert all(p.grad is None for p in m.params)


def _save_with_moments(m):
    """A checkpoint whose optimizer holds moments for every parameter (one fake step)."""
    for p in m.params:
        p.grad = 1e-3 * torch.randn_like(p)
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    return m.save_checkpoint(7)


def test_the_moments_map_onto_the_encoder_by_name(M, tmp_path):
    m = _lite(tmp_path)
    path = _save_with_moments(m)
    opt, mom = M.adam_from_checkpoint(m, path)
    enc, _ = M.encoder_of(m)
    assert set(mom["moments"]) == {n for n, _ in enc.named_parameters()}
    for n, p in enc.named_parameters():
        st = m.optimizer.state[p]
        assert torch.equal(opt.state[p]["exp_avg"], st["exp_avg"].float())
        assert torch.equal(opt.state[p]["exp_avg_sq"], st["exp_avg_sq"].float())


def test_the_whole_ring_decoder_reads_a_carried_posture_and_its_control_does_not(M):
    g = torch.Generator().manual_seed(0)
    lat = torch.cumsum(0.05 * torch.randn(3000, 4, generator=g), dim=0)
    P = lat @ torch.randn(4, 182, generator=g)
    Z = P @ (torch.randn(182, 192, generator=g) / 182 ** 0.5) + 0.05 * torch.randn(3000, 192, generator=g)
    good, lam = M.long_content_best(Z, P, folds=10, gap=50)
    ctl = M.long_content(Z, P, folds=10, gap=50, lam=lam, shift=1500)
    noise, _ = M.long_content_best(torch.randn(3000, 192, generator=g), P, folds=10, gap=50)
    assert good > 0.8 and ctl < 0.2 and noise < 0.05, (good, ctl, noise)


# ---------------------------------------------------------------- the calibration, version 6
def _cal_row(step, ratio, run="seg", alert_deaf=None, jitter=None, spread=None, posture=0.07, content=0.4,
             posture_only=None, posture_share=0.25, fatigue=0.15):
    """A version-6 calibrated checkpoint: ratio[body] = the window's response per unit movement; the token's jitter
    share 0.3 unless jitter is given, the posture's own share posture_share, the fatigue answer `fatigue`."""
    n = len(ratio)

    def per(x):
        return list(x) if isinstance(x, (list, tuple)) else [x] * n
    posture, content = per(posture), per(content)
    spread = per(spread) if spread is not None else [r * u for r, u in zip(ratio, posture)]
    jitter = per(jitter) if jitter is not None else [2 * 0.3 * s * s for s in spread]      # a share of 0.3
    block = dict(ratio=list(ratio), spread=spread, posture=posture, jitter=jitter,
                 jitter_share=[j / (2 * s * s) for j, s in zip(jitter, spread)], content=content,
                 content_control=[0.0] * n, posture_jitter_share=per(posture_share), fatigue_answer=per(fatigue),
                 ratio_posture_only=per(posture_only) if posture_only is not None else [0.96 * r for r in ratio])
    return dict(loaded_step=step, run=run, ratio=block, alert_deaf=list(alert_deaf or [False] * n))


def _v5_population():
    """Five pre-deploy windows that share no moment (two configurations; 281,562 -> 288,771 holds body 4 going deaf, a
    state change, not window noise), the straddling 344,525, and six post-deploy windows: bodies 1, 2, 6 and 8 deaf."""
    pre_steps = ((281562, "H84_seg30"), (288771, "H84_seg30"), (326129, "H84_seg30"), (332501, "H84_seg31"),
                 (344400, "H84_seg31"))
    pre = []
    for i, (s, run) in enumerate(pre_steps):
        r = [3.4 + 0.1 * k + 0.05 * i for k in range(8)]
        r[2] = 0.4                                                        # body 3: face down, left out
        if s == 288771:
            r[3] = 1.2                                                    # body 4 went deaf in this window
        pre.append(_cal_row(s, r, run=run, content=[0.5] * 8, posture_share=[0.12 + 0.01 * i + 0.002 * k
                                                                             for k in range(8)],
                            fatigue=[0.16 + 0.004 * i + 0.001 * k for k in range(8)]))
    straddle = [_cal_row(344525, [3.0] * 8, run="H84_seg32")]
    deaf = [True, True, False, False, False, True, False, True]
    post = []
    for i, s in enumerate((363041, 369250, 380550, 405325, 411400, 417500)):
        run = "H84_seg40_resume_2026092%d" % i if s > 400000 else "H84_seg3%d" % i
        r = [1.1 + 0.02 * i, 1.0, 0.3, 4.1, 4.2, 1.05, 4.3, 1.2]
        post.append(_cal_row(s, r, run=run, alert_deaf=deaf, content=[0.23 if d else 0.42 for d in deaf],
                             posture_share=[0.3 + 0.02 * (i % 2) + 0.003 * k for k in range(8)]))
    return pre, straddle, post


def test_a_checkpoint_whose_window_straddles_the_deploy_belongs_to_neither_population(M):
    assert M.population(344400) == "pre" and M.population(344525) == "straddle" == M.population(347791)
    assert M.population(350399) == "straddle" and M.population(350400) == "post" == M.population(350926)
    pre, straddle, post = _v5_population()
    out = M.calibration_rules(pre + straddle + post, boots=300, seed=3)
    assert out["straddling_checkpoints"] == [344525] and 344525 not in out["post_checkpoints"] + out["pre_checkpoints"]


def test_the_windows_that_do_not_overlap_are_the_units_of_every_count_and_bootstrap(M):
    """Copies of a window 1,500 steps apart are one window: they add nothing to the independent counts or to LINE."""
    pre, _s, post = _v5_population()
    rows = [dict(r, loaded_step=r["loaded_step"] + d) for r in post for d in (0, 1500, 3000)]
    kept = M.independent(rows)
    assert all(b["loaded_step"] - a["loaded_step"] >= 6000 for a, b in zip(kept, kept[1:]))
    assert len(kept) == len(post), "three copies 1,500 steps apart are one window"
    a = M.calibration_rules(pre + post, boots=300, seed=5)
    b = M.calibration_rules(pre + rows, boots=300, seed=5)
    assert a["counts_independent"] == b["counts_independent"] and a["LINE"] == b["LINE"]
    assert b["counts_all_post"]["readings"] == 3 * a["counts_all_post"]["readings"], "reported apart"


def test_the_calibration_rules_version_6(M):
    """LINE: the median ratio of the independent pre-deploy windows (the live level); DEAF: version 4's tolerance bound
    on the ratio, kept out of sample; the noise in log units from pre-deploy pairs of one configuration, median-based so
    a body that changed state inside a pair (body 4, 281,562 -> 288,771) is not window noise (the root-mean-square
    estimate beside it is inflated by it: the control); K6's fall; K2's line (version 6) on the token's jitter share
    minus the posture's, its noise from the pre- AND the post-deploy pairs; the content control's and the fatigue
    answer's noise; SKIP; the pre-deploy state; the counts."""
    pre, straddle, post = _v5_population()
    out = M.calibration_rules(pre + straddle + post, boots=400, seed=3)
    assert out["version"] == 6 and out["pre_independent"] == [281562, 288771, 326129, 332501, 344400]
    flat = sorted(v for r in pre for b, v in enumerate(r["ratio"]["ratio"]) if b != 2)
    assert out["LINE"] == math.floor(100 * flat[len(flat) // 2]) / 100.0, "the median, rounded down"
    assert out["line_rule"]["bootstrap_5"] <= out["line_rule"]["median"] <= out["line_rule"]["bootstrap_95"]
    assert out["DEAF"] <= math.floor(100 * out["deaf_rule"]["percentile"]) / 100.0 < out["LINE"]
    assert out["deaf_false_out_of_sample"]["windows"] == 5
    wn = out["window_noise"]
    assert wn["pairs"] == [[332501, 344400], [281562, 288771]] and wn["changes"] == 14 and wn["units"] == "ln ratio"
    assert wn["sigma_rms"] > 3 * wn["sigma"], "the state change inflates the root-mean-square estimate only"
    assert wn["sigma_hi"] >= wn["sigma"] and wn["k6_fall"] == pytest.approx(2.326 * 2 ** 0.5 * wn["sigma"])
    sd = out["share_diff"]
    diffs = [0.3 - v for r in pre + straddle + post for b, v in enumerate(r["ratio"]["posture_jitter_share"]) if b != 2]
    assert sd["largest"] == pytest.approx(max(diffs)) and sd["noise"]["sigma"] > 0
    assert out["K2_share_diff"] == pytest.approx(sd["largest"] + 2.326 * 2 ** 0.5 * sd["noise"]["sigma"])
    pre_only = M.noise_of_changes(M.pair_changes(M.noise_pairs(pre + post, "pre"), "share_diff", log=False),
                                  __import__("random").Random(1), 50)
    assert sd["noise"]["changes"] > pre_only["changes"], "the post-deploy pairs are in K2's noise"
    assert "K2_share" not in out and out["jitter_share"]["largest"] == pytest.approx(0.3)
    assert out["content_noise"]["sigma"] == pytest.approx(0.0, abs=1e-12), "the content did not change in the pairs"
    fn = out["fatigue_noise"]
    assert fn["sigma"] > 0 and fn["sigma_hi"] >= fn["sigma"] and fn["changes"] == 14, "A1 (vi)'s and LS's margin"
    assert out["SKIP"] == [3] and out["excluded"] == "body 3"
    assert out["pre_deploy"]["step"] == 344400 and out["pre_deploy"]["ratio"] == pre[-1]["ratio"]["ratio"]
    c = out["counts_independent"]
    assert c["alert_deaf"] == 24 and c["alert_deaf_under_deaf"] == 24 and c["alert_live_under_deaf"] == 0
    assert out["post_levels"]["live_median"] == pytest.approx(4.2) and out["post_levels"]["deaf"] == 24
    assert out["posture_only_over_plain"]["median_log"] == pytest.approx(math.log(0.96), abs=1e-6)
    assert "note" in M.calibration_rules(post), "no pre-deploy checkpoint: no line"


def test_a_version_4_row_serves_the_ratio_rules(M):
    """The round-7 calibrate job's rows (version 4) carry the response, the movement, the ratio and the jitter in each
    penalty's readings (none of them depends on the penalty) and the posture content at 200 (the content control's
    penalty): reading_of takes them, and computes the jitter share."""
    rd = dict(r2=[0.5, 0.2], r2_control=[0.0, 0.0], spread=[0.3, 0.1], posture=[0.08, 0.07], ratio=[3.75, 1.43],
              jitter=[0.05, 0.004])
    row = dict(loaded_step=1, readings={"20.0": dict(rd, r2=[0.6, 0.3]), "200.0": rd})
    assert M.reading_of(row) == [3.75, 1.43] and M.reading_of(row, "content") == [0.5, 0.2]
    assert M.reading_of(row, "jitter_share") == pytest.approx([0.05 / 0.18, 0.004 / 0.02])
    assert M.reading_of(row, "ratio_posture_only") is None
    assert M.judged_values(dict(row, readings={"200.0": dict(rd, ratio=[3.0, 0.0])})) == [(0, 3.0)], \
        "a ratio of 0 (no movement) has no logarithm and is not judged"


def test_the_rules_on_h84s_committed_readings(M):
    """The version-6 rules on the round-7 calibrate job's readings of H84's 22 checkpoints (record section 43): the live
    level 3.75 (every body but body 3 read 3.21-4.40 at 344,400), DEAF 2.86, one window's noise 0.117 in ln ratio (a
    reading wanders 12 in 100 of its size between windows that share no moment), and every body the deaf-sense alert
    names deaf after the deploy reads under DEAF; the promise holds out of sample on average. Those rows carry neither
    the posture's jitter share nor the fatigue answer, so version 6's K2 line and fatigue noise wait for the calibrate
    job (and say so)."""
    rows = json.loads(H84_ROWS.read_text(encoding="utf-8"))["rows"]
    out = M.calibration_rules(rows)
    assert out["LINE"] == 3.75 and out["DEAF"] == 2.86
    wn = out["window_noise"]
    assert wn["sigma"] == pytest.approx(0.1174, abs=5e-4) and wn["k6_fall"] == pytest.approx(0.386, abs=2e-3)
    assert out["deaf_false_out_of_sample"]["mean"] <= 0.10
    c = out["counts_independent"]
    assert c["alert_deaf"] == 18 and c["alert_deaf_under_deaf"] == 18
    assert out["jitter_share"]["largest"] == pytest.approx(0.548, abs=1e-3)
    assert out["K2_share_diff"] is None and any("K2" in n for n in out["notes"]), \
        "the committed rows carry no posture jitter share: version 6's K2 line waits for the calibrate job"
    assert out["fatigue_noise"]["sigma"] is None, "nor any fatigue answer"
    assert out["post_levels"]["live_median"] == pytest.approx(4.04, abs=0.01)
    assert out["pre_deploy"]["ratio"][0] == pytest.approx(3.63, abs=0.01)


def _saved_lite(tmp_path):
    m = _lite(tmp_path)
    return m, _save_with_moments(m)


def test_the_switch_test_loads_the_flagged_build_with_a_line(M, monkeypatch):
    """EM loads the checkpoint twice, with and without the flag. The lesson refuses the flag without its lines: the
    flagged load carries a stand-in line and DEAF (version 6) and the skipped body (EM compares weights and tokens
    only), and the learner builds."""
    from mimo_bridge.tools import gof_probe_harness as H
    seen = []
    monkeypatch.setattr(H, "load_frozen", lambda ck, threads=2, arg_overrides=None: seen.append(arg_overrides))
    M.load_model("ck", repair=True)
    M.load_model("ck", repair=False)
    M.load_model("ck", repair=True, line=3.75, skip=(), deaf=2.86)
    over = seen[0]
    assert seen[1] is None and over["proprio_token_repair"] == "within_body"
    assert 1.0 <= over["proprio_token_deaf"] < over["proprio_token_line"] <= 20.0 and over["proprio_token_skip"] == "3"
    assert seen[2]["proprio_token_line"] == 3.75 and seen[2]["proprio_token_skip"] is None
    assert seen[2]["proprio_token_deaf"] == 2.86
    assert "proprio_token_lambda" not in over and "proprio_token_still" not in over
    torch.set_num_threads(1)
    from mimo_bridge.lem_model import LemLite
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True, body_layout="trunk_v1",
                **over)
    assert m.proprio_within is not None and m.proprio_within.skip == (3,)


def _pace_module():
    spec = importlib.util.spec_from_file_location("proprio_token_pace_check", MEAS.parent / "proprio_token_pace_check.py")
    pace_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pace_mod)
    return pace_mod


def _pace_encoder_and_window(seed=0):
    """A small encoder of the production's shape (Linear, LayerNorm, GELU, Dropout; then Linear) and three bodies: two
    whose postures walk beside a large constant, one skipped (body 3)."""
    from mimo_bridge import proprio_token as PT
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(312, 32, bias=False), torch.nn.LayerNorm(32), torch.nn.GELU(),
                              torch.nn.Dropout(0.0), torch.nn.Linear(32, 16, bias=False))
    g = torch.Generator().manual_seed(seed + 1)
    cols = PT.posture_columns(312)
    X = 2.0 * torch.randn(3, 1, 312, generator=g).expand(3, PT.WINDOW, 312).clone()
    walk = torch.cumsum(0.05 * torch.randn(3, PT.WINDOW, 4, generator=g), dim=1)
    X[..., cols] += walk @ (0.05 * torch.randn(4, len(cols), generator=g))
    moments = {k: dict(exp_avg=1e-4 * torch.randn_like(v), exp_avg_sq=torch.full_like(v, 1e-6), step=100.0,
                       decay=v.dim() >= 2, weight_decay=0.0) for k, v in mlp.named_parameters()}
    return mlp, X, moments


def test_the_pace_of_the_lesson_is_its_first_order_change_and_leaves_the_encoder_as_it_was(M):
    """proprio_token_pace_check.lesson_pace (version 6): the lesson's per-update step through Adam's saved moments, each
    body's first-order change of ln ratio, checked on a finite step; the encoder is left as it was; the skipped body is
    no target; the capped pace is never faster; the step per weight in learning rates per input field; the updates to
    the line are ln(line / ratio) / pace; with the ring's moments, the residual path's, VICReg's and production's own
    per-update effects."""
    from mimo_bridge import proprio_token as PT
    pace_mod = _pace_module()
    mlp, X, moments = _pace_encoder_and_window()
    before = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
    X_all = X.repeat(1, 2, 1)
    out = pace_mod.lesson_pace(mlp, X, moments, dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8), line=PT.LINE_MAX,
                               skip=(3,), sigma_log_hi=0.17, steps=(1, 10), X_all=X_all,
                               episode=torch.zeros(3, X_all.shape[1], dtype=torch.int32), clip=0.01, n_senses=6)
    assert all(torch.equal(before[k], v) for k, v in mlp.state_dict().items()), "the encoder is left as it was"
    assert out["targets"] == [1, 2], "every body under the line but the skipped one"
    for b in (0, 1):
        pace = out["pace_per_update"][b]
        assert pace > 0 and out["finite_change"]["1"]["log_ratio"][b] == pytest.approx(pace, rel=0.25), out
        assert out["updates_to_line"][str(b + 1)] == round(math.log(PT.LINE_MAX / out["ratio"][b]) / pace)
        assert 0 < out["pace_per_update_capped"][b] <= pace, "the lesson's own gradient only slows a weight"
    assert out["a1_line"] == pytest.approx(PT.LINE_MAX * math.exp(-2 * 0.17 / 3 ** 0.5))
    assert set(out["other_pressures_per_update"]) == {"residual", "vicreg", "production"}
    assert all(len(v) == 3 for v in out["other_pressures_per_update"].values())
    first = out["tensors"]["0.weight"]
    assert set(first["by_input_field"]) == set(PT.INPUT_FIELDS)
    assert first["lesson_step_in_lr"]["rms"] == pytest.approx(PT.STRENGTH, rel=1e-3), \
        "a uniform second moment: every weight's step averages STRENGTH learning rates"
    assert len(out["content"]) == 3 and len(out["finite_change"]["10"]["content_change"]) == 3


def test_the_pace_jobs_a1_line_is_version_6s_settled_line_when_deaf_is_given(M):
    """Record section 47: version 6 moved A1's line to where a lifted body settles, DEAF x exp(-2 SIGMA_LOG_HI / sqrt 3)
    (the watcher's acceptance), but the pace job kept version 5's LINE x exp(...) (3.08 where the watcher judges 2.35).
    With DEAF given the pace job's A1 line is DEAF's and its updates count to it; without DEAF it stays version 5's.
    Fails on 039acbc38 (its A1 line reads LINE's)."""
    from mimo_bridge import proprio_token as PT
    pace_mod = _pace_module()
    mlp, X, moments = _pace_encoder_and_window()
    hyper = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    start = pace_mod.lesson_path(mlp, X, moments, hyper, line=PT.LINE_MAX, skip=(3,), chunks=0)["rows"][0]["ratio"]
    deaf = 1.5 * max(start[:2])
    out = pace_mod.lesson_pace(mlp, X, moments, hyper, line=PT.LINE_MAX, skip=(3,), sigma_log_hi=0.17, steps=(1,),
                               deaf=deaf)
    assert out["targets"] == [1, 2], "both walking bodies read under DEAF"
    a1 = deaf * math.exp(-2 * 0.17 / 3 ** 0.5)
    assert out["a1_line"] == pytest.approx(a1) and out["a1_line_from"] == "DEAF"
    assert out["a1_line"] != pytest.approx(PT.LINE_MAX * math.exp(-2 * 0.17 / 3 ** 0.5)), "not version 5's line"
    for b in (0, 1):
        pace = out["pace_per_update"][b]
        assert out["updates_to_a1_line"][str(b + 1)] == max(0, round(math.log(a1 / out["ratio"][b]) / pace))
    # a DEAF under a body's ratio puts that body over A1's line: no target, nothing to count
    low = pace_mod.lesson_pace(mlp, X, moments, hyper, line=PT.LINE_MAX, skip=(3,), sigma_log_hi=0.17, steps=(1,),
                               deaf=0.5 * min(start[:2]))
    assert low["targets"] == [] and low["updates_to_a1_line"] == {}
    old = pace_mod.lesson_pace(mlp, X, moments, hyper, line=PT.LINE_MAX, skip=(3,), sigma_log_hi=0.17, steps=(1,))
    assert old["a1_line"] == pytest.approx(PT.LINE_MAX * math.exp(-2 * 0.17 / 3 ** 0.5))
    assert old["a1_line_from"].startswith("LINE")


def test_the_lessons_path_reaches_the_line_and_stops_pushing_there(M):
    """proprio_token_pace_check.lesson_path: the lesson alone, re-aimed every chunk; each target rises to the line, a
    body at or over it is no longer pushed (the hinge), and the encoder is put back."""
    from mimo_bridge import proprio_token as PT
    pace_mod = _pace_module()
    mlp, X, moments = _pace_encoder_and_window(seed=2)
    before = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
    start = pace_mod.lesson_path(mlp, X, moments, dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8), line=PT.LINE_MAX,
                                 skip=(3,), chunks=0)["rows"][0]["ratio"]
    line = max(PT.LINE_MIN, 2.0 * max(start[:2]))
    out = pace_mod.lesson_path(mlp, X, moments, dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8), line=line, skip=(3,),
                               strength=PT.STRENGTH_MAX, chunks=40, chunk=50)
    assert all(torch.equal(before[k], v) for k, v in mlp.state_dict().items()), "the encoder is put back"
    rows = out["rows"]
    assert rows[0]["ratio"] == pytest.approx(start) and rows[0]["pushed"] == [1, 2]
    for b in (1, 2):
        at = out["updates_to_line"][str(b)]
        assert at is not None, (b, [r["ratio"] for r in rows])
        after = [r for r in rows if r["updates"] > at]
        assert all(b not in r["pushed"] for r in after if r["ratio"][b - 1] >= line)
    assert "3" not in out["updates_to_line"], "the skipped body is not followed"


def test_the_paths_hinge_engages_under_deaf_and_its_hold_is_the_lessons_on_the_pushed_bodies_only(M, monkeypatch):
    """Version 6's path (proprio_token_pace_check.lesson_path): the hinge pushes only a body under DEAF (the other body
    under the line is left alone); the hold's line is read as the lesson reads it (version 7: the median fatigue answer
    ON THE POSTURE-ONLY ROWS of the live bodies at the path's start, proprio_token.hold_line_of; here the only live
    body's; version 6 read its answer on the real rows), its reading is the lesson's formula (_hold_parts against
    proprio_token), and it acts only on a body the path has pushed: a hold with the line under every answer holds body 1
    (pushed) and never body 2 (never pushed). hold=False (version 5's lesson) holds nothing. Fails on 67335783f (the
    real rows' answer)."""
    from mimo_bridge import proprio_token as PT
    monkeypatch.setattr(PT, "LINE_MIN", 1e-6)                  # the typing guard, set under this toy's scale
    pace_mod = _pace_module()
    mlp, X, moments = _pace_encoder_and_window(seed=4)
    # limb speeds that change over the window: the posture-only rows (speed and fatigue held at their mean) then read
    # another token than the real rows, so the two answers differ
    vel = [j for j in range(312) if j % 12 in PT.VELOCITY_FIELDS]
    X[..., vel] += 0.05 * torch.randn(3, PT.WINDOW, len(vel), generator=torch.Generator().manual_seed(44))
    hyper = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    r = pace_mod.lesson_path(mlp, X, moments, hyper, line=PT.LINE_MAX, skip=(3,), chunks=0)["rows"][0]["ratio"]
    lo, hi = sorted(r[:2])
    low = r.index(lo)
    deaf, line = (lo + hi) / 2, min(PT.LINE_MAX, 2.0 * hi)
    out = pace_mod.lesson_path(mlp, X, moments, hyper, line=line, skip=(3,), chunks=3, deaf=deaf)
    rows = out["rows"]
    assert rows[0]["engaged"] == [low + 1] and rows[0]["pushed"] == [low + 1], "only the body under DEAF"
    assert all(r_["pushed_ever"] == [low + 1] for r_ in rows)
    live = 1 - low
    xl = PT.posture_only(X)[live:live + 1]
    with torch.no_grad():
        zl = pace_mod._tokens(mlp, xl)
        want_live = float(PT.fatigue_change(zl, pace_mod._tokens(mlp, PT.fatigue_stepped(xl)))[0] / PT.response(zl)[0])
    assert out["fatigue_line"] == pytest.approx(want_live, rel=1e-5), \
        "the hold's line: the live body's answer on the posture-only rows at the start (the only live body)"
    assert out["fatigue_answer_posture_only_start"][live] == pytest.approx(want_live, rel=1e-5)
    assert out["fatigue_line"] != pytest.approx(rows[0]["fatigue_answer"][live], rel=1e-3), "not the real rows'"
    assert all(set(r_["held"]) <= {low + 1} for r_ in rows), "never a body the path did not push"
    xp = PT.posture_only(X)[low:low + 1]
    with torch.no_grad():
        z = pace_mod._tokens(mlp, xp)
        want = float(PT.fatigue_change(z, pace_mod._tokens(mlp, PT.fatigue_stepped(xp)))[0] / PT.response(z)[0])
        ans, over, _lnf = pace_mod._hold_parts(mlp, xp, PT.response(z)[0], 0.5 * want)
    assert ans == pytest.approx(want, rel=1e-5) and over == pytest.approx(math.log(2.0), rel=1e-5)
    free = pace_mod.lesson_path(mlp, X, moments, hyper, line=line, skip=(3,), chunks=3, deaf=deaf, hold=False)
    assert all(r_["held"] == [] for r_ in free["rows"]) and free["fatigue_line"] is None


def test_the_held_out_reading_reads_the_second_half_through_the_moved_encoder(M, monkeypatch):
    """The third review's SEMANTIC (proprio_token_pace_check.held_out_reading): the path trained on the first half of
    the window, read on the second, which it never trained on, through the unmoved and the moved encoder (the gain is
    the moved ratio over the unmoved); the gain-matched content divides that gain out; the heading reading turns the
    other judged bodies' rows to each target's heading; the encoder is put back."""
    from mimo_bridge import proprio_token as PT
    monkeypatch.setattr(PT, "LINE_MIN", 1e-6)                  # the typing guard, set under this toy's scale
    pace_mod = _pace_module()
    mlp, X, moments = _pace_encoder_and_window(seed=6)
    before = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
    hyper = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    second = X[:, X.shape[1] // 2:]
    with torch.no_grad():
        u = PT.movement(M.posture_of(second))
        r = PT.ratio_of(PT.response(pace_mod._tokens(mlp, second)), u).tolist()
    lo, hi = sorted(r[:2])
    low = r.index(lo)
    out = pace_mod.held_out_reading(mlp, X, moments, hyper, line=2.0 * hi, deaf=(lo + hi) / 2, skip=(3,), chunks=4)
    assert all(torch.equal(before[k], v) for k, v in mlp.state_dict().items()), "the encoder is put back"
    assert out["unmoved"]["ratio"] == pytest.approx(r, abs=1e-4), "the readings are printed to 4 decimals"
    assert out["gain"][low] == pytest.approx(out["moved"]["ratio"][low] / out["unmoved"]["ratio"][low], rel=1e-2)
    assert out["path_updates"] == 400 and out["moved"]["ratio"] != out["unmoved"]["ratio"], "read through the moved one"
    assert len(out["moved"]["content_gain_matched"]) == 3 and len(out["moved"]["fatigue_answer"]) == 3
    assert set(out["heading"]) == {str(low + 1)} and set(out["heading"][str(low + 1)]["others"]) == {str(2 - low)}
    # fix round 2 (the re-review's finding 7): the posture-only answers the hold's line is judged on, both encoders, and
    # the bodies the held-out path pushed
    assert out["unmoved"]["fatigue_answer_posture_only"] == pytest.approx(
        pace_mod.posture_only_answers(mlp, second.contiguous()), abs=1e-4)
    assert len(out["moved"]["fatigue_answer_posture_only"]) == 3
    assert out["moved"]["fatigue_answer_posture_only"] != out["unmoved"]["fatigue_answer_posture_only"]
    assert out["path_pushed_ever"] == [low + 1]


def test_the_pace_jobs_hold_line_judges_the_posture_only_answers(M):
    """The re-review's finding 7 (fix round 2): section 45's pace line (1), every pushed body's fatigue answer at the
    path's end at most F + 2 x FATIGUE_SIGMA_HI / sqrt 3, is judged on the POSTURE-ONLY answers, the rows F is read on
    (on H84 the pushed bodies answer 3-6 in 100 more there than on the plain rows, so a plain-row answer passes more
    easily), in sample (the path's last row) and held out (the encoder moved on the first half, read on the second,
    against the first half's F). Here body 1's posture-only answer ends over the line while its plain answer is under it:
    missed; an unmeasured noise, an unread pushed body or no pushed body never passes. Fails on ded74f88d (no
    hold_line_verdict) and on the planted defect that judges the plain answers."""
    pace_mod = _pace_module()
    F, fsig = 0.167, 0.035
    line = F + 2 * fsig / 3 ** 0.5                                      # 0.2074

    def path(p1):
        return dict(fatigue_line=F, rows=[dict(updates=0), dict(
            updates=1100, pushed_ever=[1, 8], fatigue_answer=[0.20, 0.1, 0.05, 0.16, 0.16, 0.1, 0.16, 0.15],
            fatigue_answer_posture_only=[p1, None, None, None, None, None, None, 0.16])])
    held = dict(fatigue_line=0.170, path_pushed_ever=[1, 8],
                moved=dict(fatigue_answer=[0.19, 0.1, 0.05, 0.16, 0.16, 0.1, 0.16, 0.15],
                           fatigue_answer_posture_only=[0.20, 0.1, 0.05, 0.16, 0.16, 0.1, 0.16, 0.16]))
    over = pace_mod.hold_line_verdict(path(0.215), held, fsig)
    assert over["rows"] == "posture-only" and over["in_sample"]["line"] == pytest.approx(line)
    assert over["in_sample"]["bodies"] == {"1": 0.215, "8": 0.16} and over["in_sample"]["met"] is False
    assert over["met"] is False, "0.215 over the line on the posture-only rows (0.20, under it, on the plain rows)"
    ok = pace_mod.hold_line_verdict(path(0.20), held, fsig)
    assert ok["met"] is True and ok["held_out"]["met"] is True and ok["held_out"]["line"] == pytest.approx(
        0.170 + 2 * fsig / 3 ** 0.5)
    held_over = dict(held, moved=dict(held["moved"], fatigue_answer_posture_only=[0.22] + [0.1] * 7))
    assert pace_mod.hold_line_verdict(path(0.20), held_over, fsig)["met"] is False, "held out over its line"
    assert pace_mod.hold_line_verdict(path(0.20), held, None)["met"] is None, "an unmeasured noise passes nothing"
    assert pace_mod.hold_line_verdict(path(None), held, fsig)["met"] is None, "an unread pushed body passes nothing"
    nobody = dict(fatigue_line=F, rows=[dict(updates=0, pushed_ever=[], fatigue_answer_posture_only=[None] * 8)])
    got = pace_mod.hold_line_verdict(nobody, None, fsig)
    assert got["met"] is None and "no body was pushed" in got["why"]


def test_the_pace_job_maps_the_saved_moments_onto_the_encoders_own_names():
    pace_mod = _pace_module()
    mlp = torch.nn.Sequential(torch.nn.Linear(12, 4, bias=False), torch.nn.LayerNorm(4))
    saved = {"mlp." + k: dict(step=1.0) for k, _ in mlp.named_parameters()}
    with pytest.raises(KeyError):
        [saved[k] for k, _ in mlp.named_parameters()]              # the defect: the saved names miss
    got = pace_mod.named_moments(mlp, saved)
    assert set(got) == {k for k, _ in mlp.named_parameters()}
    with pytest.raises(ValueError, match="missing"):
        pace_mod.named_moments(mlp, {"mlp.0.weight": dict(step=1.0)})


def test_a_ring_extract_serves_the_pace_job_only_for_its_own_checkpoint(tmp_path):
    """The light path (proprio_mechanism/ring_extract.py): the pace job rebuilds the inputs from an extract of the same
    checkpoint's ring, and refuses an extract of another."""
    pace_mod = _pace_module()
    ach = torch.zeros(2, 30, 26, 10, dtype=torch.float16)
    ach[..., 3:7] = torch.nn.functional.normalize(torch.randn(2, 30, 26, 4), dim=-1).half()
    ach[..., 9] = 1.0
    f = tmp_path / "ring.pt"
    torch.save(dict(step=7, checkpoint=str(tmp_path / "ckpt_step_7"), achieved=ach,
                    episode=torch.zeros(2, 30, dtype=torch.int32), nf_raw=None), f)
    X, ep = pace_mod.lazy_ring(tmp_path / "ckpt_step_7", ring_file=f)
    assert X.shape == (2, 30, 312) and ep.shape == (2, 30)
    assert torch.allclose(X.view(2, 30, 26, 12)[..., 0:4], ach[..., 3:7].float())
    with pytest.raises(SystemExit, match="extracted from"):
        pace_mod.lazy_ring(tmp_path / "ckpt_step_8", ring_file=f)


def _gap_module():
    spec = importlib.util.spec_from_file_location("proprio_token_input_gap", MEAS.parent / "proprio_token_input_gap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_input_gap_job_reads_real_against_rebuilt_rows_and_its_controls_hold(M, tmp_path):
    """proprio_token_input_gap.py (review 2 MAJOR: the calibration reads inputs rebuilt from the ring's poses, the lesson
    the real ones). On a lived trajectory whose real velocity carries a jitter the one-step difference does not and
    whose capacity drifts: the posture numbers agree to float16's resolution; putting the real velocity and capacity back
    reproduces the real reading; the real rows read twice give one number; its ratio readings give version 5's gap. The
    control: a rebuild with a planted defect (positions not made wrist-relative) fails the posture check."""
    from mimo_bridge import proprio_token as PT
    gapm = _gap_module()
    m, _path = _saved_lite(tmp_path)
    sd, norm = M.live_state(m)
    g = torch.Generator().manual_seed(9)
    steps, H = 6000, 26
    t = torch.arange(steps, dtype=torch.float32)[:, None, None]
    pos = 0.3 + 0.05 * torch.randn(1, H, 3, generator=g) + 0.02 * torch.sin(2 * math.pi * t / torch.tensor(
        [300.0, 470.0, 810.0]).view(1, 1, 3) + torch.rand(1, H, 3, generator=g) * 6.28)          # world cm / 200
    quat = torch.nn.functional.normalize(torch.randn(1, H, 4, generator=g) + 0.1 * torch.sin(t / 200.0), dim=-1)
    poses = torch.zeros(steps, H, 10)
    poses[..., 0:3], poses[..., 3:7], poses[..., 9] = pos, quat, 1.0
    poses16 = poses.to(torch.float16)
    real = torch.zeros(steps, H, 12)
    real[..., 0:4] = quat
    real[..., 4:7] = pos - pos[:, 0:1]
    vel = torch.zeros(steps, H, 3)
    vel[1:] = (pos[1:] - pos[:-1]) * (200.0 / 0.05 / 500.0)
    vel += 0.01 * torch.randn(steps, H, 3, generator=g)                  # the physics' jitter
    real[..., 7:10] = vel
    real[..., 10] = 0.9 - 0.04 * t[..., 0] / steps                       # the capacity drifts over the window
    real[..., 11] = torch.arange(H, dtype=torch.float32) / 25.0
    real = real.reshape(steps, H * 12)
    rows = real[torch.arange(steps - 1 - (PT.WINDOW - 1) * PT.SAMPLE_EVERY, steps, PT.SAMPLE_EVERY)]
    rebuilt_all = gapm.rebuild(poses16.float(), rows[-1])
    idx = torch.arange(steps - 1 - (PT.WINDOW - 1) * PT.SAMPLE_EVERY, steps, PT.SAMPLE_EVERY)
    windows = gapm.variants(rows, rebuilt_all[idx])
    assert set(windows) == {"real", "rebuilt", "real_velocity", "real_capacity", "real_both"}
    out = gapm.gap_readings(sd, norm, windows, [60.0, 200.0])
    assert out["controls_met"] == dict(posture=True, velocity_and_capacity_account_for_the_gap=True, repeatable=True), \
        out["controls"]
    assert out["gap"]["200.0"]["r2"] == pytest.approx(out["readings"]["200.0"]["real"]["r2"]
                                                      - out["readings"]["200.0"]["rebuilt"]["r2"])
    assert out["readings"]["60.0"]["real"]["ratio"] == out["readings"]["200.0"]["real"]["ratio"] > 0, \
        "the ratio does not depend on the penalty: the watcher reads its gap from any"
    broken = windows["rebuilt"].clone().view(PT.WINDOW, H, 12)
    broken[..., 4:7] += poses16.float()[idx][:, 0:1, 0:3]
    bad = gapm.gap_readings(sd, norm, dict(windows, rebuilt=broken.view(PT.WINDOW, -1)), [60.0])
    assert bad["controls_met"]["posture"] is False


def test_the_area_under_the_curve_and_its_standard_error(M):
    assert M.auc([0.5, 0.6], [0.1, 0.2]) == (1.0, pytest.approx(0.0, abs=1e-12))
    a_, se = M.auc([0.5, 0.2, 0.6], [0.3, 0.1])
    assert a_ == pytest.approx(5 / 6) and 0 < se < 0.5
    assert M.auc([], [0.1]) == (None, None)


def test_the_calibration_reads_the_checkpoints_own_encoder_over_the_whole_window_as_the_measure_job_does(
        M, tmp_path, monkeypatch):
    """Section 32: the calibration and the measure job read one token (the checkpoint's own encoder over the lesson's
    whole window, the card's arithmetic), never the ring's stored tokens. Version 5: the calibration's ratio block is the
    measure job's reading of the same window on the same weights."""
    from mimo_bridge import proprio_token as PT
    m, path = _saved_lite(tmp_path)
    rep, nf_raw = _stand_in_ring(m, N=6000)
    monkeypatch.setattr(M, "companion_replay", lambda ck: (rep, nf_raw))
    row = M.calibrate_checkpoint(str(path), lambda r: None)
    ring = M.Ring(rep, nf_raw)
    X = ring.lesson_window()
    assert row["window"]["samples"] == PT.WINDOW == X.shape[1]
    block = {k: v for k, v in row["ratio"].items() if k != "heading_transfer"}
    assert block == M.content_readings(m, ring), "one number, one window"
    assert set(row["ratio"]) == {"spread", "posture", "ratio", "spread_posture_only", "ratio_posture_only", "jitter",
                                 "jitter_share", "posture_jitter_share", "fatigue_answer", "content", "content_control",
                                 "heading_transfer"}
    P = M.posture_of(X)
    assert row["ratio"]["posture_jitter_share"] == pytest.approx(PT.jitter_share(P).tolist(), rel=1e-6)
    at = row["ratio"]["heading_transfer"]["at_body"]
    assert set(at) == {"1", "2"} and set(at["1"]["others"]) == {"2"}, "each judged body read at the others' heading"
    Z = M.lesson_tokens(*M._state_and_x(M.saved_state(path), X))
    for lam in M.LAMBDAS:
        assert row["readings"][str(lam)]["r2"] == M.window_readings(Z, X, lam)["r2"], "the record keeps version 4's"
    assert len(row["alert_deaf"]) == 2 and row["alert_line"] == 0.20


def test_the_alert_reads_the_collapse_watchs_50_s_window_not_the_lessons(M):
    from mimo_bridge import proprio_token as PT
    m = _lite()
    rep, nf_raw = _stand_in_ring(m, N=6000)
    ring = M.Ring(rep, nf_raw)
    idx = ring.watch_index(5999)
    assert len(idx) == 200 != PT.WINDOW and int(idx[0]) == 5999 - 995 and int(idx[1] - idx[0]) == 5
    assert all(int(ring.watch_index(e)[0]) >= 0 for e in M.watch_ends(ring.N))
    Zs = ring.tokens[:, :, ring.pi]
    want = [float(sorted(float(Zs[b, ring.watch_index(e)].std(0).mean()) for e in M.watch_ends(ring.N))[2:4][0] * 0.5
                  + sorted(float(Zs[b, ring.watch_index(e)].std(0).mean()) for e in M.watch_ends(ring.N))[2:4][1] * 0.5)
            for b in range(ring.B)]
    assert M.alert_spreads(ring) == pytest.approx(want)


def test_the_card_encoding_is_the_models_own_encoder_in_bf16(M, tmp_path):
    F = torch.nn.functional
    m, path = _saved_lite(tmp_path)
    enc, encode = M.encoder_of(m)
    x = 0.3 * torch.randn(64, 312)
    with torch.no_grad():
        f32 = encode(x)
        card = M.card_encode(*M.live_state(m)[:1], x, M.live_state(m)[1])
        h = x.float()
        for mod in enc.mlp:
            if isinstance(mod, torch.nn.Linear):
                h = F.linear(h.bfloat16(), mod.weight.bfloat16(), None if mod.bias is None else mod.bias.bfloat16())
            elif isinstance(mod, torch.nn.LayerNorm):
                h = F.layer_norm(h.float(), mod.normalized_shape, mod.weight.float(), mod.bias.float(), eps=mod.eps)
            elif isinstance(mod, torch.nn.GELU):
                h = F.gelu(h)
        norm = M.live_state(m)[1]
        walked = F.layer_norm(h.float(), (h.shape[-1],), None if norm is None else norm[0].float(),
                              None if norm is None else norm[1].float(), eps=1e-5)
    assert torch.equal(walked, card), "the module's own layers, the card's casts"
    rms = float((f32 - card).square().mean().sqrt())
    assert 1e-4 < rms < 0.01, rms
    sd, norm = M.saved_state(path)
    assert torch.equal(M.card_encode(sd, x, norm), card), "the saved weights are the live ones"
    n_linear = sum(isinstance(mod, torch.nn.Linear) for mod in enc.modules())
    assert n_linear == sum(1 for k, v in sd.items() if k.endswith(".weight") and v.dim() == 2)


def test_every_refit_arm_starts_from_the_same_encoder_and_puts_it_back(M, tmp_path, monkeypatch):
    from mimo_bridge import proprio_token as PT
    monkeypatch.setitem(M.LINES, "eval_every", 5)
    monkeypatch.setitem(M.LINES, "eval_windows", 2)
    monkeypatch.setitem(M.LINES, "clip_calibration", 2)
    m = _lite(tmp_path)
    _prime_norm(m)
    path = _save_with_moments(m)
    rep, nf_raw = _stand_in_ring(m, N=6000)
    ring = M.Ring(rep, nf_raw)
    enc, _ = M.encoder_of(m)
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    with pytest.raises(ValueError, match="LINE"):
        M.refit_arm(m, ring, path, "R", 1, 11, lambda: None)
    with pytest.raises(ValueError, match="DEAF"):
        M.refit_arm(m, ring, path, "R", 1, 11, lambda: None, line=PT.LINE_MAX)
    arms = dict(C=M.refit_arm(m, ring, path, "C", 10, 11, lambda: None, line=PT.LINE_MAX, skip="2"))
    arms["R"] = M.refit_arm(m, ring, path, "R", 10, 11, lambda: None, line=PT.LINE_MAX, skip="2",
                            deaf=PT.LINE_MAX - 0.01, reference=arms["C"]["rows"])
    after = enc.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "the encoder is put back after each arm"
    for a in arms.values():
        assert [r["update"] for r in a["rows"]] == [0, 5, 10]
        assert set(a["rows"][0]["ratios"]) >= {"tactile", "audio", "proprioception"}
        assert len(a["clip_factors"]) == len(before) and all(f > 0 for f in a["clip_factors"])
    c0 = arms["C"]["rows"][0]["content"]
    assert arms["R"]["rows"][0]["content"]["ratio"] == c0["ratio"], "both arms start from the same encoder"
    assert set(c0) == {"spread", "posture", "ratio", "spread_posture_only", "ratio_posture_only", "jitter",
                       "jitter_share", "posture_jitter_share", "fatigue_answer", "content", "content_control"}
    r0 = arms["R"]["rows"][0]["content"]
    assert r0["gain_ref"] == c0["ratio"] and r0["content_gain_matched"] == pytest.approx(c0["content"], abs=1e-6), \
        "arm R divides its gain by arm C's ratio at the same update: at update 0 the gain is 1"
    assert arms["R"]["rows"][-1]["content"]["gain_ref"] == arms["C"]["rows"][-1]["content"]["ratio"]
    X = M.ramped_window(ring)
    assert c0["ratio"] == M.content_readings(m, ring, X=X)["ratio"], "the arms read the ramped window on its token"
    assert arms["R"]["rows"][-1]["engaged"] is not None, "arm R's lesson ran its hinge"
    assert arms["R"]["lesson_share"] is not None, "arm R applied the lesson after the scaled gradient"
    assert arms["R"]["rows"][-1]["content"]["ratio"][0] != arms["C"]["rows"][-1]["content"]["ratio"][0]
    assert arms["R"]["attributed"] is not None and len(arms["R"]["attributed"]) == ring.B
    assert arms["R"]["lesson_step_lr"] and arms["C"]["attributed"] is None


def test_an_arm_on_the_shared_factors_draws_its_seeds_moments_and_arm_r_records_the_hold(M, tmp_path, monkeypatch):
    """Version 7 (record section 47 and its fix round): an arm given the checkpoint's shared clip factors
    (shared_clip_factors, seeded clip_seed = 11, version 6's arms C and R's seed) draws past the calibration moments and
    then the same training moments as version 6's arm of its seed, so seeded 11 it repeats version 6's arm C reading for
    reading (the plumbing; since the review no judged pair runs seed 11: stage 1 runs seeds 13-20); heldout_at reads the
    held-out next-moment ratios at those updates only; a factor list of another length is refused; the factors' draw puts
    every parameter's gradient flag back. ARM R'S ROWS CARRY THE HOLD'S GAUGES at each evaluation and the counts of the
    updates in which the hold and the push acted on each body, and the arm returns the totals (the review's finding 3);
    arm C's rows carry none. Fails on 67335783f (no hold readings in arm R)."""
    monkeypatch.setitem(M.LINES, "eval_every", 5)
    monkeypatch.setitem(M.LINES, "eval_windows", 2)
    monkeypatch.setitem(M.LINES, "clip_calibration", 2)
    m = _lite(tmp_path)
    _prime_norm(m)
    path = _save_with_moments(m)
    rep, nf_raw = _stand_in_ring(m, N=6000)
    ring = M.Ring(rep, nf_raw)
    v6 = M.refit_arm(m, ring, path, "C", 10, 11, lambda: None)
    for p in m.params:
        p.requires_grad_(True)
    factors = M.shared_clip_factors(m, ring, path)
    assert all(p.requires_grad for p in m.params), "the flags are put back"
    assert factors == pytest.approx(v6["clip_factors"], rel=1e-6), "version 6's arm C drew its factors on seed 11"
    v7 = M.refit_arm(m, ring, path, "C", 10, 11, lambda: None, factors=factors, heldout_at={0, 10})
    assert [r["update"] for r in v7["rows"]] == [r["update"] for r in v6["rows"]] == [0, 5, 10]
    for a, b in zip(v6["rows"], v7["rows"]):
        assert b["content"]["ratio"] == pytest.approx(a["content"]["ratio"], rel=1e-5), "the same moments, the same steps"
    assert v7["clip_factors_shared"] is True and v6["clip_factors_shared"] is False
    assert [r["ratios"] is not None for r in v7["rows"]] == [True, False, True]
    other = M.refit_arm(m, ring, path, "C", 10, 13, lambda: None, factors=factors, heldout_at={0, 10})
    assert other["rows"][-1]["content"]["ratio"] != v7["rows"][-1]["content"]["ratio"], "another seed, other moments"
    with pytest.raises(ValueError, match="shared clip factors"):
        M.refit_arm(m, ring, path, "C", 1, 11, lambda: None, factors=factors[:-1])
    from mimo_bridge import proprio_token as PT
    r13 = M.refit_arm(m, ring, path, "R", 10, 13, lambda: None, line=PT.LINE_MAX, deaf=PT.LINE_MAX - 0.01,
                      reference=other["rows"], factors=factors, heldout_at={0, 10})
    assert "hold" not in other["rows"][0] and other["held_updates"] is None, "arm C runs no lesson"
    hold0, hold_end = r13["rows"][0]["hold"], r13["rows"][-1]["hold"]
    assert hold0["held_updates"] == [0] * ring.B and hold0["lesson_updates"] == 0, "counted from update 0"
    assert hold_end["held_updates"] == r13["held_updates"] and hold_end["pushed_updates"] == r13["pushed_updates"]
    assert len(r13["held_updates"]) == ring.B and r13["lesson_updates"] == 10 and max(r13["pushed_updates"]) == 10
    assert set(hold_end) >= {"held", "hold", "push", "fatigue_line", "fatigue_answer_posture_only", "pushed"}


def test_arm_r_counts_the_holds_updates_on_a_pushed_body_once_a_live_body_gives_its_line(M, tmp_path, monkeypatch):
    """The re-review's finding 6 (a planted defect no test caught: arm R's hold counts never incremented). The plumbing
    test above engages every body at update 0, so no body is live, the hold's line F is unmeasured, the hold never acts
    and its counts hold for zero. Here one body is engaged (pushed) and the other stays live and never pushed (the hinge
    held so: the stand-in re-fit moves both lite bodies' ratios 25 in 100 in 5 updates, so no DEAF keeps them apart), and
    F is read from the live body's posture-only answer (the lesson's own hold_line_of, scaled here to a millionth of it
    so the hold acts on the pushed body at every update after the start). Arm R counts the updates in which the hold
    acted on the pushed body, 9 of 10 (the first update reads the start and F), and none on the live body; the push acted
    on the pushed body in all 10."""
    from mimo_bridge import proprio_token as PT
    monkeypatch.setitem(M.LINES, "eval_every", 5)
    monkeypatch.setitem(M.LINES, "eval_windows", 2)
    monkeypatch.setitem(M.LINES, "clip_calibration", 2)
    m = _lite(tmp_path)
    _prime_norm(m)
    path = _save_with_moments(m)
    rep, nf_raw = _stand_in_ring(m, N=6000)
    ring = M.Ring(rep, nf_raw)
    pushed, live = 0, 1

    def engage_one(self, ratio, has):
        self.engaged = [b == pushed for b in range(int(ratio.shape[0]))]
        return torch.tensor(self.engaged, dtype=torch.bool, device=ratio.device)
    monkeypatch.setattr(PT.WithinBody, "engage", engage_one)
    real = PT.hold_line_of
    seen = []

    def scaled(answers, never):
        f = real(answers, never)
        seen.append((list(answers), list(never), f))
        return None if f is None else 1e-6 * f
    monkeypatch.setattr(PT, "hold_line_of", scaled)
    for p in m.params:
        p.requires_grad_(True)
    factors = M.shared_clip_factors(m, ring, path)
    arm = M.refit_arm(m, ring, path, "R", 10, 13, lambda: None, line=PT.LINE_MAX, deaf=PT.LINE_MAX - 0.01,
                      factors=factors, heldout_at={0, 10})
    assert arm["pushed_updates"][pushed] == 10 and arm["pushed_updates"][live] == 0, arm["pushed_updates"]
    assert arm["held_updates"][pushed] == 9 and arm["held_updates"][live] == 0, arm["held_updates"]
    start = next(c for c in seen if c[2] is not None)
    assert start[1][live] and not start[1][pushed], "F is read from the live body only"
    hold_end = arm["rows"][-1]["hold"]
    assert hold_end["fatigue_line"] == pytest.approx(1e-6 * start[0][live], rel=1e-6)
    assert hold_end["held_updates"] == arm["held_updates"] and hold_end["held"][pushed] is True
    assert arm["rows"][0]["hold"]["held_updates"] == [0, 0], "counted from update 0"


def test_the_clip_factor_is_read_from_the_runs_training_records(M, tmp_path, monkeypatch):
    run = tmp_path / "runs" / "H84_x"
    (run / "checkpoints" / "ckpt_step_500").mkdir(parents=True)
    lines = [json.dumps(dict(type="training", training_step=s, avg_grad_norm=n))
             for s, n in ((100, 50.0), (200, 400.0), (300, 200.0), (600, 9999.0))]
    (run / "replay_training_stats.jsonl").write_text("\n".join(lines + ['{"type": "mimo", "step": 5}']),
                                                     encoding="utf-8")
    monkeypatch.setattr(M.R, "LIVE_RUNS", tmp_path / "runs")
    out = M.clip_factor(str(run / "checkpoints" / "ckpt_step_500"))
    assert out["records"] == 3 and out["median_norm"] == 200.0 and out["factor"] == pytest.approx(1 / 200.0)
    assert "3 of the 20 training records" in out["note"], "fewer than 20: the factor says what it rests on"


def test_the_clip_factor_walks_the_parent_runs_for_its_twenty_records(M, tmp_path, monkeypatch):
    """The review's finding 6: the run that saved 415,075 holds 1 training record at or before it and seg37's run 4
    before 380,550, so the clip factor rested on 1-4 records. It now walks back through the parent runs as the watcher's
    before_reports does (a run owns the steps after the checkpoint it resumed; its parent the steps up to that
    checkpoint, never the parent's records after it) until grade_records are read, and says how many and from which
    runs. Here 3 records are needed: the child's own (step 500: 60) and the parent's two newest up to its resume point
    (steps 400 and 300: 100 and 200; the parent's record after the resume point, 9,999, is another branch): the median
    100. Fails on 67335783f (the child's one record: 1/60)."""
    monkeypatch.setitem(M.LINES, "grade_records", 3)
    runs = tmp_path / "runs"
    parent, child = runs / "H84_parent", runs / "H84_child"
    for run in (parent, child):
        (run / "checkpoints").mkdir(parents=True)
    (parent / "checkpoints" / "ckpt_step_400").mkdir()
    (child / "checkpoints" / "ckpt_step_500").mkdir()

    def stats(run, recs):
        (run / "replay_training_stats.jsonl").write_text("\n".join(
            json.dumps(dict(type="training", training_step=s, avg_grad_norm=n)) for s, n in recs), encoding="utf-8")
    stats(parent, ((100, 50.0), (300, 200.0), (400, 100.0), (450, 9999.0)))
    stats(child, ((500, 60.0),))
    (child / "continuation_receipt.json").write_text(json.dumps(dict(
        parent_run=str(parent), checkpoint=str(parent / "checkpoints" / "ckpt_step_400"))), encoding="utf-8")
    monkeypatch.setattr(M.R, "LIVE_RUNS", runs)
    out = M.clip_factor(str(child / "checkpoints" / "ckpt_step_500"))
    assert out["records"] == 3 and out["median_norm"] == 100.0 and out["factor"] == pytest.approx(1 / 100.0), out
    assert [(x["run"], x["records"]) for x in out["runs"]] == [("H84_child", 1), ("H84_parent", 3)]
    assert out["steps"] == [300, 500] and "note" not in out
    # a parent named by args.json's resume serves the same way
    (child / "continuation_receipt.json").unlink()
    (child / "args.json").write_text(json.dumps(dict(resume=str(parent / "checkpoints" / "ckpt_step_400"))),
                                     encoding="utf-8")
    assert M.clip_factor(str(child / "checkpoints" / "ckpt_step_500"))["median_norm"] == 100.0


def test_one_updates_effect_agrees_with_a_finite_step(M, tmp_path):
    m = _lite(tmp_path)
    path = _save_with_moments(m)
    rep, nf_raw = _stand_in_ring(m, N=6000)
    ring = M.Ring(rep, nf_raw)
    enc, encode = M.encoder_of(m)
    params = list(enc.parameters())
    for p in m.params:
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    from mimo_bridge import proprio_token as PT
    lam = PT.CONTROL_LAMBDA
    pre = M.preconditioner(m, path)
    cg = M.content_gradients(m, ring, lam)
    o = M.next_moment_terms(m, ring, 1500, encode)
    term = sum(o["terms"].values())
    g = torch.autograd.grad(term, params)
    clip = 0.5
    first = M.effect(g, params, pre, clip, cg[0])
    X = ring.lesson_window()

    def body1():
        with torch.no_grad():
            z = encode(X[0]).reshape(1, X.shape[1], -1)
            return float(PT.posture_content(z, M.posture_of(X[:1]), lam=lam)[0])
    before = body1()
    keep = [p.detach().clone() for p in params]
    k = 1e-2
    with torch.no_grad():
        for p, gi in zip(params, g):
            p.add_(-clip * k * gi * pre[id(p)])
    after = body1()
    with torch.no_grad():
        for p, kp in zip(params, keep):
            p.copy_(kp)
    assert (after - before) / k == pytest.approx(first, rel=0.2, abs=1e-6)


def test_the_grading_probe_accounts_for_the_whole_step(M):
    lin = torch.nn.Linear(3, 2)
    params = list(lin.parameters())
    pre = {id(p): torch.ones_like(p) for p in params}
    cg = [torch.ones_like(p) for p in params]
    probe = M.GradeProbe(params, pre, 1.0, cg)
    x = torch.randn(4, 3)
    a = lin(x).square().mean()
    b = 0.5 * lin(x).abs().mean()
    probe.terms.update(a=a, b=b)
    probe.before_backward(None, a + b)
    st = probe.steps[0]
    assert set(st["terms"]) == {"a", "b"} and st["unattributed"] == pytest.approx(0.0, abs=1e-6)
    assert probe.terms == {}, "the terms are cleared for the next step"


def test_a_stop_mid_checkpoint_keeps_every_stage_already_read(M, tmp_path, monkeypatch):
    import types

    def fake_measure(ck, updates, save, **kw):
        row = dict(checkpoint=ck, loaded_step=415000, EM=dict(met=True))
        save(row)
        row["arms"] = {}
        partial = row["arms"]["C"] = dict(arm="C", seed=11, rows=[], partial=True)
        partial["rows"].append(dict(update=0))
        save(row)
        partial["rows"].append(dict(update=50))
        save(row)
        raise KeyboardInterrupt                   # a kill: nothing after this line runs

    monkeypatch.setattr(M, "measure_checkpoint", fake_measure)
    args = types.SimpleNamespace(cmd="measure", output=str(tmp_path), updates=400, refit_steps=1, checkpoints=["ck1"],
                                 line=3.0, strength=None, steps=12)
    with pytest.raises(KeyboardInterrupt):
        M.run(args)
    doc = json.loads((tmp_path / "proprio_token_measure.json").read_text(encoding="utf-8"))
    row = doc["rows"][0]
    assert row["EM"] == {"met": True} and row["checkpoint"] == "ck1"
    assert row["arms"]["C"]["partial"] is True and row["arms"]["C"]["rows"] == [{"update": 0}, {"update": 50}]
    assert not M.finished(row["arms"]["C"]) and M.finished(dict(arm="C", rows=[{"update": 400}]))
    assert M.rule_ls(doc["rows"], 3.0)["met"] is False and M.rule_ls(doc["rows"], 3.0)["checkpoints"] == []


def test_measure_and_grade_take_their_settings_from_the_version_6_calibration(M, tmp_path, monkeypatch):
    """The pause queues calibrate and measure together: measure reads LINE, DEAF, SKIP and the content and fatigue
    noise from the version-6 calibrate result and passes them to every stage; a version-5 result (no fatigue answer, no
    share difference), a version-4 result (whose LINE is an R2) and a result that set no LINE are refused, and so is a
    --line that differs; grade reads version 4's content at CONTROL_LAMBDA unless told another."""
    import types
    from mimo_bridge import proprio_token as PT
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(dict(rules=dict(version=6, LINE=3.75, DEAF=2.86, SKIP=[3],
                                              window_noise=dict(sigma=0.117, sigma_hi=0.171, k6_fall=0.386),
                                              content_noise=dict(sigma=0.05, sigma_hi=0.069),
                                              fatigue_noise=dict(sigma=0.01, sigma_hi=0.015),
                                              pre_checkpoints=[288771]))), encoding="utf-8")
    seen = {}

    def fake(name):
        def stage(ck, *a, **kw):
            seen[name] = kw
            return dict(checkpoint=ck, loaded_step=415000)
        return stage
    monkeypatch.setattr(M, "measure_checkpoint", fake("measure"))
    monkeypatch.setattr(M, "grade_checkpoint", fake("grade"))
    for cmd in ("measure", "grade"):
        args = types.SimpleNamespace(cmd=cmd, output=str(tmp_path / cmd), updates=400, refit_steps=1,
                                     checkpoints=["ck1"], line=None, lam=None, strength=None, steps=12,
                                     from_calibration=str(cal) if cmd == "measure" else None)
        assert M.run(args) == 0
    doc = json.loads((tmp_path / "measure" / "proprio_token_measure.json").read_text(encoding="utf-8"))
    assert doc["line"] == 3.75 and doc["deaf"] == 2.86 and doc["skip"] == [3]
    assert doc["calibration"]["calibration"] == str(cal) and doc["calibration"]["sigma_log_hi"] == 0.171
    assert seen["measure"]["line"] == 3.75 and seen["measure"]["skip"] == "3" and seen["measure"]["deaf"] == 2.86
    assert seen["grade"]["lam"] == PT.CONTROL_LAMBDA
    cal_read = M.from_calibration(cal)
    assert cal_read["content_sigma_hi"] == 0.069 and cal_read["fatigue_sigma_hi"] == 0.015
    for version, rules in ((4, dict(LINE=0.32, LAMBDA=200.0, STILL=0.0834)), (5, dict(LINE=3.75, DEAF=2.86))):
        old = tmp_path / ("v%d.json" % version)
        old.write_text(json.dumps(dict(rules=dict(rules, version=version))), encoding="utf-8")
        with pytest.raises(SystemExit, match="not version 6"):
            M.run(types.SimpleNamespace(cmd="measure", output=str(tmp_path / "x"), updates=1, refit_steps=1,
                                        checkpoints=["ck1"], line=None, lam=None, strength=None, steps=1,
                                        from_calibration=str(old)))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(dict(rules=dict(version=6, note="no pre-deploy checkpoint read"))), encoding="utf-8")
    with pytest.raises(SystemExit, match="sets no LINE"):
        M.run(types.SimpleNamespace(cmd="measure", output=str(tmp_path / "y"), updates=1, refit_steps=1,
                                    checkpoints=["ck1"], line=None, lam=None, strength=None, steps=1,
                                    from_calibration=str(empty)))
    with pytest.raises(SystemExit, match="differs from the calibration"):
        M.run(types.SimpleNamespace(cmd="measure", output=str(tmp_path / "z"), updates=1, refit_steps=1,
                                    checkpoints=["ck1"], line=3.0, lam=None, strength=None, steps=1,
                                    from_calibration=str(cal)))


def _arm(ratio_start, ratio_end, other=1.0, U=400, posture_end=None, content_end=None, gm_end=None, fatigue=None,
         fatigue_end=None):
    """An arm's first and last evaluation (version 6's readings). posture_end: the posture-only ratio at the end
    (default 0.96 x the ratio); content_end: the content control at the end (default 0.3); gm_end: the gain-matched
    content at the end (default the content); fatigue: the fatigue answer at update 0 (default 0.15 for every body);
    fatigue_end: at the end (default the same)."""
    rows = []
    n = len(ratio_start)
    f0 = list(fatigue) if fatigue is not None else [0.15] * n
    for u, rr in ((0, ratio_start), (U, ratio_end)):
        end = u == U
        content = list(content_end) if end and content_end is not None else [0.3] * n
        rows.append(dict(update=u, content=dict(
            ratio=list(rr), posture=[0.07] * n, spread=[0.07 * x for x in rr],
            ratio_posture_only=(list(posture_end) if end and posture_end is not None else [0.96 * x for x in rr]),
            content=content, content_gain_matched=(list(gm_end) if end and gm_end is not None else content),
            fatigue_answer=(list(fatigue_end) if end and fatigue_end is not None else f0)),
            ratios={"vision": other, "tactile": 1.0, "proprioception": 1.1}))
    return dict(rows=rows)


def test_the_rules_ls_and_nf(M):
    line, deaf = 3.0, 2.0
    csig, fsig = 0.069, 0.015
    # bodies 1 and 2 deaf (under DEAF), body 3 (excluded) deaf too, bodies 4 and 5 live: their fatigue answers at update
    # 0 (0.15 and 0.12) give the hold's line, the larger
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    good = dict(loaded_step=400000, arms=dict(C=_arm(start, [1.0, 1.2, 0.3, 4.0, 3.9], fatigue=f0),
                                              R=_arm(start, [1.6, 1.8, 0.3, 4.1, 4.0], fatigue=f0,
                                                     fatigue_end=[0.16, 0.14, 0.05, 0.15, 0.12]),
                                              C2=_arm(start, [1.01, 1.21, 0.3, 4.0, 3.9], other=1.01, fatigue=f0)))

    def ls_(rows, *a, **kw):
        kw.setdefault("content_sigma_hi", csig)
        kw.setdefault("fatigue_sigma_hi", fsig)
        return M.rule_ls(rows, *a, **kw)
    ls = ls_([good], line, deaf)
    ck = ls["checkpoints"][0]
    assert ls["met"] and ck["targets"] == [1, 2], ck
    b1 = ck["bodies"][0]
    assert b1["fatigue_line"] == pytest.approx(0.15), "the live bodies' largest answer at update 0, not their median"
    assert b1["answers_fatigue_as_live"] and b1["fatigue_answer"] == pytest.approx(0.16), \
        "0.16 is inside the line plus its noise (0.15 + 2 x 0.015 / sqrt 3 = 0.167)"
    assert b1["pace_per_update"] == pytest.approx(math.log(1.6) / 400)
    assert b1["horizon"] == pytest.approx(math.log(3.0) / (math.log(1.6) / 400))
    b2 = ck["bodies"][1]
    assert ck["A1_horizon"] == pytest.approx(2 * max(b1["horizon"], b2["horizon"]))
    within = dict(good, arms=dict(good["arms"], R=_arm(start, [1.005, 1.205, 0.3, 4.0, 3.9], fatigue=f0)))
    assert not ls_([within], line, deaf)["met"], "a gain inside the re-fit's noise is not a lift"
    costs = dict(good, arms=dict(good["arms"], R=_arm(start, [1.6, 1.8, 0.3, 3.0, 3.9], fatigue=f0)))
    got = ls_([costs], line, deaf)
    assert not got["met"] and got["checkpoints"][0]["bodies"][3]["kept"] is False, "no body may lose response"
    # the lift must answer the posture: a lift the posture-only ratio does not share fails
    slow = dict(good, arms=dict(good["arms"], R=_arm(start, [1.6, 1.8, 0.3, 4.1, 4.0], fatigue=f0,
                                                     posture_end=[0.96, 1.15, 0.3, 3.9, 3.8])))
    got = ls_([slow], line, deaf)
    assert not got["met"] and got["checkpoints"][0]["bodies"][0]["answers_posture"] is False
    # and carry it: a lift whose gain-matched content falls past its margin (the third review's planted token: its
    # plain content held, 0.019-0.060 with the gain divided out) moves without posture information
    planted = dict(good, arms=dict(good["arms"], R=_arm(start, [1.6, 1.8, 0.3, 4.1, 4.0], fatigue=f0,
                                                        gm_end=[0.1, 0.3, 0.3, 0.3, 0.3])))
    got = ls_([planted], line, deaf)
    assert not got["met"] and got["checkpoints"][0]["bodies"][0]["follows_posture"] is False
    assert got["checkpoints"][0]["bodies"][1]["follows_posture"] is True
    # and answer fatigue as the live bodies do: a lifted body whose answer ends over the hold's line plus its noise
    # fails (version 5's path: 0.26-0.35 against the live bodies' 0.15-0.17)
    tired = dict(good, arms=dict(good["arms"], R=_arm(start, [1.6, 1.8, 0.3, 4.1, 4.0], fatigue=f0,
                                                      fatigue_end=[0.30, 0.14, 0.05, 0.15, 0.12])))
    got = ls_([tired], line, deaf)
    assert not got["met"] and got["checkpoints"][0]["bodies"][0]["answers_fatigue_as_live"] is False
    assert not ls_([good], line, deaf, fatigue_sigma_hi=None)["met"], "an unmeasured fatigue noise decides nothing"
    # a skipped body is neither a target nor judged
    got = ls_([good], line, deaf, skip="1")
    assert got["checkpoints"][0]["targets"] == [2] and got["checkpoints"][0]["bodies"][0]["judged"] is False
    # with no target body nothing was lifted, and that is never "met"
    quiet = ls_([good], line, deaf=0.5)
    assert quiet["checkpoints"][0]["targets"] == [] and quiet["state"] == "no targets" and not quiet["met"]
    # without DEAF the targets are the bodies under the line
    assert ls_([good], line)["checkpoints"][0]["targets"] == [1, 2]
    good_att = dict(good, arms=dict(good["arms"], R=dict(_arm(start, [1.6, 1.8, 0.3, 4.1, 4.0], fatigue=f0),
                                                          attributed=[0.5, 0.4, 0.0, 0.0, 0.0])))
    b1 = ls_([good_att], line, deaf)["checkpoints"][0]["bodies"][0]
    assert b1["attributed_over_measured"] == pytest.approx(0.5 / math.log(1.6))
    nf = M.rule_nf([good])
    assert nf["met"] and nf["checkpoints"][0]["factor"] == pytest.approx(max(1.02, 1.01))
    worse = dict(good, arms=dict(good["arms"], R=_arm(start, [1.6, 1.8, 0.3, 4.1, 4.0], other=1.2, fatigue=f0)))
    assert not M.rule_nf([worse])["met"]


# the no-lesson wander each pair draws (every body's ln ratio at the last update over its start; the live heading group
# of 380,550 wandered 0.36-0.54 in 400 updates) and the jitter of the paired gain between pairs
_WANDER = [0.05, -0.30, 0.20, -0.15, 0.40, 0.00, -0.45, 0.12]
_JITTER = [0.01, -0.02, 0.015, -0.01, 0.0, 0.02, -0.015, 0.005]


def _pairs_row(start, gains, fatigue=None, r_fatigue_end=None, r_gm_end=None, other=1.0, r_other=None, U=400,
               stage=1, seed0=13, held=None, pushed=None, attributed=None, declared=None):
    """A version-7 row: one pair per entry of gains (gains[s][b] = ln R_s - ln C_s of body b at the last update); arm C
    of pair s ends at start x exp(_WANDER[s]), arm R at that times exp(gains[s][b]); the posture-only ratios 0.96 times
    the ratios; shared clip factors 0.1 and 0.2 against production's clip factor 1/200 from 20 records. Each pair is
    tagged with its stage and seeded seed0 + s; held/pushed/attributed: arm R's per-pair lists per body (the hold's and
    the push's update counts, the lesson's attribution). `other`/`r_other`: the other senses' held-out ratio in the two
    arms (a number, or one per pair)."""
    pairs = []
    for s, g in enumerate(gains):
        c_end = [x * math.exp(_WANDER[s % len(_WANDER)]) for x in start]
        r_end = [c * math.exp(gb) for c, gb in zip(c_end, g)]
        o_c = other[s] if isinstance(other, (list, tuple)) else other
        o_r = o_c if r_other is None else (r_other[s] if isinstance(r_other, (list, tuple)) else r_other)
        C = dict(_arm(start, c_end, U=U, fatigue=fatigue, other=o_c), arm="C", seed=seed0 + s)
        R = dict(_arm(start, r_end, U=U, fatigue=fatigue, fatigue_end=r_fatigue_end, gm_end=r_gm_end, other=o_r),
                 arm="R", seed=seed0 + s)
        for key, v in (("held_updates", held), ("pushed_updates", pushed), ("attributed", attributed)):
            if v is not None:
                R[key] = list(v[s])
        pairs.append(dict(seed=seed0 + s, stage=stage, C=C, R=R))
    return dict(loaded_step=400000, pairs=pairs, pairs_declared=declared or len(gains), clip_factors_shared=[0.1, 0.2],
                clip_record=dict(factor=1 / 200.0, records=20, runs=[dict(run="H84_x", records=20)]))


def _with_stage2(row, gains, seed0=21, **kw):
    """row with a stage-2 set of pairs appended (seeds 21-28)."""
    start = row["pairs"][0]["C"]["rows"][0]["content"]["ratio"]
    more = _pairs_row(start, gains, stage=2, seed0=seed0, **kw)["pairs"]
    return dict(row, pairs=row["pairs"] + more)


def test_the_t_quantiles_are_students(M):
    """Rule LS version 7's t bounds: the one-sided 95-in-100 table (df 1-30) and the 97.5-in-100 table every look of the
    two-stage rule judges at, and the expansion above 30, against scipy's quantiles. The unit job must show this test
    passed: scipy is imported here, never skipped (the review's finding 11; its import committed 1.49 GiB, which the
    unit job's memory need counts)."""
    import scipy.stats as stats
    for level in (0.95, 0.975):
        for df in range(1, 61):
            assert M.t_one_sided(df, level) == pytest.approx(float(stats.t.ppf(level, df)), abs=1e-3), (level, df)
    assert M.t_one_sided_95(7) == 1.895 and M.t_one_sided(7) == 2.365 and M.t_one_sided(15) == 2.131
    assert M.LINES["ls_look_level"] == 0.975 and M.t_one_sided(0) == math.inf
    with pytest.raises(ValueError, match="levels"):
        M.t_one_sided(7, 0.9)


def test_rule_ls_version_7_judges_the_paired_gain_against_its_spread_over_the_pairs(M):
    """Record section 47 and its fix round: rule LS version 7 judges each body's paired gain (ln R_s - ln C_s at the
    last update, the two arms of pair s drawing the same moments) against its spread over the pairs, with Student's
    one-sided bounds at 97.5 in 100 at each look (t(7) = 2.365 on stage 1's 8 pairs); the no-lesson wander, which version
    6 read as the noise from one pair, decides nothing; no pair count "that would resolve" a target is printed (the
    review's finding 2: it invited adding pairs after the result). Fails on 67335783f (its bound is 95 in 100, it prints
    pairs_to_resolve and names the gap single_draw_gap_v6)."""
    line, deaf, csig, fsig = 3.0, 2.0, 0.069, 0.015
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]

    def gains(targets, live=0.0, t_jit=1.0, l_jit=0.5):
        return [[targets + t_jit * j, targets - t_jit * j, 0.0, live + l_jit * j, live - l_jit * j] for j in _JITTER]

    def ls_(row, **kw):
        kw.setdefault("content_sigma_hi", csig)
        kw.setdefault("fatigue_sigma_hi", fsig)
        kw.setdefault("sigma_log", 0.117)
        kw.setdefault("k6_fall_log", 0.386)
        return M.rule_ls_pairs([row], line, kw.pop("deaf", deaf), None, **kw)
    assert not hasattr(M, "pairs_to_resolve") and "ls_pairs_max" not in M.LINES
    assert M.LINES["ls_seed0"] == 13 and M.LINES["ls_seed0_stage2"] == 21 and M.LINES["clip_seed"] == 11
    # a lift of 0.3 that every pair repeats within 0.02, under a no-lesson wander of up to 0.45: met at look 1
    good = _pairs_row(start, gains(0.3), fatigue=f0)
    ls = ls_(good)
    ck = ls["checkpoints"][0]
    assert ls["state"] == "met" and ls["met"] and ck["targets"] == [1, 2] and ck["pairs"] == 8, ck
    assert [lk["look"] for lk in ck["looks"]] == [1] and ck["t"] == 2.365 and ck["stage1_seeds"] == list(range(13, 21))
    b1 = ck["bodies"][0]
    g1 = [0.3 + j for j in _JITTER]
    m, sd = sum(g1) / 8, (sum((x - sum(g1) / 8) ** 2 for x in g1) / 7) ** 0.5
    assert b1["lift"] == "lifted" and b1["posture_only_lift"] == "lifted" and b1["state"] == "lifted"
    assert b1["log_gain"] == pytest.approx(m) and b1["gain_sd"] == pytest.approx(sd)
    assert b1["gain_lower"] == pytest.approx(m - 2.365 * sd / 8 ** 0.5) and b1["decided_at_look"] == 1
    assert b1["gains"] == pytest.approx(g1) and b1["pair_seeds"] == list(range(13, 21))
    w = [math.log(math.exp(x)) for x in _WANDER]
    assert b1["wander_sd"] == pytest.approx((sum((x - sum(w) / 8) ** 2 for x in w) / 7) ** 0.5)
    assert b1["no_lesson_gap_pairs_1_2"] == pytest.approx(abs(_WANDER[0] - _WANDER[1])), "reported only"
    assert "single_draw_gap_v6" not in b1 and "pairs_to_resolve" not in b1
    amp = ck["amplification"]
    assert amp["low"] == pytest.approx(20.0) and amp["high"] == pytest.approx(40.0) and amp["clip_records"] == 20
    assert all(x["kept"] for x in ck["bodies"] if x["judged"])
    # the same lift with its sign flipping between pairs (mean 0.01): unresolved at look 1, stage 2 due, never met
    flip = [[g, g, 0.0, 0.0, 0.0] for g in (0.3, -0.25, 0.28, -0.3, 0.1, -0.1, 0.2, -0.15)]
    got = ls_(_pairs_row(start, flip, fatigue=f0))
    ck = got["checkpoints"][0]
    assert got["state"] == "unresolved" and not got["met"] and ck["bodies"][0]["lift"] == "unresolved"
    assert ck["stage2"]["due"] and not ck["stage2"]["ran"] and ck["stage2"]["seeds"] == list(range(21, 29))
    assert got["stage2_due"] == [400000]
    # a target the lesson lowers in every pair: not lifted, missed
    lower = [[-0.1 + j, 0.3 + j, 0.0, 0.0, 0.0] for j in _JITTER]
    got = ls_(_pairs_row(start, lower, fatigue=f0))
    assert got["state"] == "missed" and got["checkpoints"][0]["bodies"][0]["lift"] == "not lifted"
    # a live body the lesson lowers in every pair by more than one window's noise: lost, missed with both targets lifted
    lost = [[0.3 + j, 0.3 - j, 0.0, -0.2 + j, 0.0] for j in _JITTER]
    got = ls_(_pairs_row(start, lost, fatigue=f0))
    assert got["state"] == "missed" and got["checkpoints"][0]["bodies"][3]["lost"] is True
    assert got["checkpoints"][0]["bodies"][0]["state"] == "lifted"
    # the lift carries the posture and answers fatigue as the live bodies do (version 6's lines on the pairs' means)
    got = ls_(_pairs_row(start, gains(0.3), fatigue=f0, r_gm_end=[0.1, 0.3, 0.3, 0.3, 0.3]))
    assert got["state"] == "missed" and got["checkpoints"][0]["bodies"][0]["follows_posture"] is False
    got = ls_(_pairs_row(start, gains(0.3), fatigue=f0, r_fatigue_end=[0.30, 0.10, 0.05, 0.15, 0.12]))
    assert got["state"] == "missed" and got["checkpoints"][0]["bodies"][0]["answers_fatigue_as_live"] is False
    assert not ls_(good, fatigue_sigma_hi=None)["met"], "an unmeasured fatigue noise passes nothing"
    # no target: never met; the two versions' rows never cross
    assert ls_(good, deaf=0.5)["state"] == "no targets"
    v6 = dict(loaded_step=400000, arms=dict(C=_arm(start, start), R=_arm(start, start), C2=_arm(start, start)))
    assert M.rule_ls_pairs([v6], line, deaf)["checkpoints"] == [] and M.rule_ls([good], line, deaf)["checkpoints"] == []


def test_the_verdict_needs_every_declared_pair_and_prints_every_pairs_gain(M):
    """The coordinator's decision E: a look needs every declared pair of its stage ("incomplete" otherwise, never a
    pass: version 7's first form judged 7 of 8 pairs on t(6)); the verdict printed for a person carries every pair's
    gain beside it (gains skewed by rare large losses read "lifted" 43 times in 100 at 8 pairs: they are read before a
    deploy). Fails on 67335783f (it judged the 7 finished pairs; it printed the rules' first 3,000 characters only)."""
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    gains = [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER]
    cut = _pairs_row(start, gains, fatigue=[0.10, 0.09, 0.05, 0.15, 0.12])
    cut["pairs"][7]["R"]["partial"] = True
    got = M.rule_ls_pairs([cut], 3.0, 2.0, None, 0.069, 0.015, 0.117, 0.386)
    ck = got["checkpoints"][0]
    assert got["state"] == "incomplete" and not got["met"] and ck["state"] == "incomplete"
    assert "7 of stage 1's 8 declared pairs" in ck["why"] and ck["bodies"] == []
    nf = M.rule_nf_pairs([cut])
    assert not nf["met"] and nf["checkpoints"][0]["state"] == "incomplete"
    whole = _pairs_row(start, gains, fatigue=[0.10, 0.09, 0.05, 0.15, 0.12])
    rules = dict(LS=M.rule_ls_pairs([whole], 3.0, 2.0, None, 0.069, 0.015, 0.117, 0.386))
    text = "\n".join(M.verdict_lines(rules))
    assert "rule LS version 7: met" in text and "checkpoint 400,000: met" in text and "look 1 on 8 pairs" in text
    assert "seed 13 %+.3f" % gains[0][0] in text and "seed 20 %+.3f" % gains[7][0] in text, "every pair's gain"
    assert "body 4 (not a target): kept" in text
    assert M.verdict_lines(dict(LS=M.rule_ls([], 3.0))) == [], "version 6's rule prints its JSON only"


def test_the_two_stage_rule_judges_look_2_on_all_sixteen_pairs_for_what_look_1_left_unresolved(M):
    """The review's finding 2 and the coordinator's decision: a checkpoint look 1 leaves unresolved gets exactly one
    stage 2 of 8 more pairs (seeds 21-28); look 2 judges the targets look 1 left unresolved on all 16 pairs at t(15) =
    2.131 (97.5 in 100), a target look 1 resolved keeps its state, every judged body's LOST is read again, and after look
    2 "unresolved" is final. Stage-2 pairs in a row whose look 1 did not read unresolved are never judged. Fails on
    67335783f (one look at 95 in 100 over every pair it finds)."""
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    kw = dict(content_sigma_hi=0.069, fatigue_sigma_hi=0.015, sigma_log=0.117, k6_fall_log=0.386)
    # body 1: a lift of 0.1 with a spread of 0.12 (unresolved on 8 pairs, lifted on 16); body 2 lifted clearly
    s1 = [0.25, -0.05, 0.2, -0.02, 0.15, 0.1, -0.05, 0.22]
    s2 = [0.12, 0.18, 0.05, 0.2, 0.16, 0.08, 0.14, 0.1]
    row1 = _pairs_row(start, [[g, 0.3 + j, 0.0, j, -j] for g, j in zip(s1, _JITTER)], fatigue=f0)
    one = M.rule_ls_pairs([row1], 3.0, 2.0, None, **kw)["checkpoints"][0]
    assert one["state"] == "unresolved" and one["bodies"][0]["lift"] == "unresolved"
    assert one["bodies"][1]["state"] == "lifted"
    both = _with_stage2(row1, [[g, 0.3 + j, 0.0, j, -j] for g, j in zip(s2, _JITTER)], fatigue=f0)
    got = M.rule_ls_pairs([both], 3.0, 2.0, None, **kw)
    ck = got["checkpoints"][0]
    assert got["state"] == "met" and [lk["look"] for lk in ck["looks"]] == [1, 2] and ck["looks"][1]["t"] == 2.131
    assert ck["bodies"][0]["decided_at_look"] == 2 and ck["bodies"][0]["pairs"] == 16
    assert ck["bodies"][1]["decided_at_look"] == 1 and ck["pairs"] == 16 and ck["stage2_seeds"] == list(range(21, 29))
    # the same stage 2 leaving body 1 on both sides of 0: unresolved after both stages is final, never a pass
    s2x = [0.3, -0.3, 0.25, -0.28, 0.2, -0.2, 0.3, -0.25]
    worse = _with_stage2(row1, [[g, 0.3 + j, 0.0, j, -j] for g, j in zip(s2x, _JITTER)], fatigue=f0)
    got = M.rule_ls_pairs([worse], 3.0, 2.0, None, **kw)
    assert got["state"] == "unresolved" and "final, never a pass" in got["checkpoints"][0]["why"]
    # a stage 2 that stopped early: incomplete
    part = _with_stage2(row1, [[g, 0.3 + j, 0.0, j, -j] for g, j in zip(s2, _JITTER)], fatigue=f0)
    part["pairs"][-1]["C"]["partial"] = True
    assert M.rule_ls_pairs([part], 3.0, 2.0, None, **kw)["state"] == "incomplete"
    # stage-2 pairs beside a look 1 that met: not judged
    good = _pairs_row(start, [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER], fatigue=f0)
    extra = _with_stage2(good, [[-0.5, -0.5, 0.0, -0.5, -0.5]] * 8, fatigue=f0)
    got = M.rule_ls_pairs([extra], 3.0, 2.0, None, **kw)
    assert got["state"] == "met" and "not judged" in got["checkpoints"][0]["note"]
    # NF: judged at each look, met only when met at both
    nf_bad2 = _with_stage2(row1, [[g, 0.3 + j, 0.0, j, -j] for g, j in zip(s2, _JITTER)], fatigue=f0, r_other=1.3)
    nf = M.rule_nf_pairs([nf_bad2])
    assert [lk["look"] for lk in nf["checkpoints"][0]["looks"]] == [1, 2]
    assert nf["checkpoints"][0]["looks"][0]["met"] and not nf["checkpoints"][0]["looks"][1]["met"] and not nf["met"]


def test_lost_has_a_size_and_a_loss_the_pairs_cannot_bound_is_printed(M):
    """The review's finding 9: a judged body is LOST when its gain's upper bound lies under 0 AND its mean under
    -SIGMA_LOG (one window's noise, 0.117); a resolved loss of 5 in 100 is not a loss. "Loss unresolved" is printed
    beside a body whose lower bound lies under -K6_FALL_LOG (0.386), deciding nothing. Without SIGMA_LOG the upper
    bound alone decides (the stricter reading). Fails on 67335783f (a loss of any size missed the checkpoint)."""
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    kw = dict(content_sigma_hi=0.069, fatigue_sigma_hi=0.015, sigma_log=0.117, k6_fall_log=0.386)
    small = [[0.3 + j, 0.3 - j, 0.0, -0.05 + 0.2 * j, 0.0] for j in _JITTER]
    got = M.rule_ls_pairs([_pairs_row(start, small, fatigue=f0)], 3.0, 2.0, None, **kw)
    b4 = got["checkpoints"][0]["bodies"][3]
    assert b4["gain_upper"] < 0 and not b4["lost"] and got["state"] == "met", "a loss under one window's noise"
    strict = M.rule_ls_pairs([_pairs_row(start, small, fatigue=f0)], 3.0, 2.0, None,
                             **dict(kw, sigma_log=None))
    assert strict["checkpoints"][0]["bodies"][3]["lost"] and strict["state"] == "missed"
    big = [[0.3 + j, 0.3 - j, 0.0, -0.2 + 0.2 * j, 0.0] for j in _JITTER]
    assert M.rule_ls_pairs([_pairs_row(start, big, fatigue=f0)], 3.0, 2.0, None, **kw)["state"] == "missed"
    wide = [[0.3 + j, 0.3 - j, 0.0, g, 0.0] for j, g in zip(_JITTER, (0.3, -0.9, 0.2, -0.7, 0.1, -0.8, 0.4, -0.6))]
    got = M.rule_ls_pairs([_pairs_row(start, wide, fatigue=f0)], 3.0, 2.0, None, **kw)
    b4 = got["checkpoints"][0]["bodies"][3]
    assert b4["loss_unresolved"] and not b4["lost"] and got["state"] == "met"
    assert "body 4 (not a target): loss unresolved" in "\n".join(M.verdict_lines(dict(LS=got)))


def test_the_horizon_comes_from_the_mean_pace_and_a_strength_raise_is_a_new_version(M):
    """The review's finding 10 and the coordinator's decision C: a target's horizon is ln(LINE / its start) over its MEAN
    paired gain per update, the lower bound's horizon printed beside it; A1's horizon is twice the largest; above 20,000
    updates the rule flags that a STRENGTH raise would be needed and that it is a new version (fresh pairs, a review),
    never a rerun of this rule. Fails on 67335783f (its horizon came from the lower bound, which raised STRENGTH up to 5
    times for any bound between 0 and 0.0133 at body 4's shortfall, and the record then ran LS again)."""
    start = [2.69, 2.8, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    kw = dict(content_sigma_hi=0.069, fatigue_sigma_hi=0.015, sigma_log=0.117, k6_fall_log=0.386)
    gains = [[0.12 + j, 0.12 - j, 0.0, j, -j] for j in _JITTER]
    ck = M.rule_ls_pairs([_pairs_row(start, gains, fatigue=f0)], 3.75, 2.86, None, **kw)["checkpoints"][0]
    b1 = ck["bodies"][0]
    assert b1["horizon"] == pytest.approx(math.log(3.75 / 2.69) / (b1["log_gain"] / 400))
    assert b1["horizon_lower"] == pytest.approx(math.log(3.75 / 2.69) / (b1["gain_lower"] / 400))
    assert b1["horizon_lower"] > b1["horizon"]
    assert ck["A1_horizon"] == pytest.approx(2 * max(x["horizon"] for x in ck["bodies"] if x["target"]))
    assert not ck["strength_raise_needed"] and ck["new_version_needed"] is None
    slow = [[0.012 + 0.01 * j, 0.012 - 0.01 * j, 0.0, j, -j] for j in _JITTER]
    ck = M.rule_ls_pairs([_pairs_row(start, slow, fatigue=f0)], 3.75, 2.86, None, **kw)["checkpoints"][0]
    assert ck["state"] == "met" and ck["A1_horizon"] > 20000 and ck["strength_raise_needed"]
    assert "a new version (fresh pairs, a review), never a rerun of this rule" in ck["new_version_needed"]


def test_nf_version_7_judges_the_paired_log_ratio_bound_on_both_sides_paired(M):
    """The review's finding 8: rule NF version 7's first form set each other sense's mean paired log ratio against the
    largest UNPAIRED no-lesson difference over the pairs (errs toward a pass). Now: each sense's paired log ratio
    ln(R_s / C_s), its one-sided 97.5-in-100 upper bound over the pairs at most ln 1.02. Here the lesson's arms read
    vision 1.5 in 100 worse on average with a spread whose bound passes 2 in 100, while the no-lesson arms wander 10 in
    100 between pairs: the first form passed it, this one does not. Fails on 67335783f."""
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    gains = [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER]
    c_other = [1.0, 1.1, 0.95, 1.05, 1.0, 1.1, 0.97, 1.02]
    r_over = [0.001, 0.03, 0.002, 0.028, 0.0, 0.03, 0.001, 0.028]
    row = _pairs_row(start, gains, other=c_other, r_other=[c * math.exp(d) for c, d in zip(c_other, r_over)])
    nf = M.rule_nf_pairs([row])
    look = nf["checkpoints"][0]["looks"][0]
    v = look["senses"]["vision"]
    assert v["mean_ratio"] < 1.02 < v["upper_ratio"] and not v["met"] and not nf["met"]
    assert look["senses"]["tactile"]["met"], "touch's arms read the same in every pair"
    same = _pairs_row(start, gains, other=c_other)
    assert M.rule_nf_pairs([same])["met"], "the lesson's arms equal the no-lesson arms: met"


def _set_ratio(row, sense, c=None, r=None):
    """`row` with `sense`'s held-out ratio set at every reading of every pair's arms C and R (c, r: one value, or one per
    pair; None as the measure job writes it when the error of predicting no change is 0)."""
    for s, p in enumerate(row["pairs"]):
        for arm, v in (("C", c), ("R", r)):
            val = v[s] if isinstance(v, (list, tuple)) else v
            for x in p[arm]["rows"]:
                x["ratios"] = dict(x["ratios"], **{sense: val})
    return row


def test_nf_reads_a_sense_no_pair_can_read_as_not_read_and_judges_the_other_senses(M, tmp_path):
    """The re-review's BLOCKER (finding 1) and the coordinator's decision: version 6's arms recorded the balance sense's
    held-out ratio as None at every reading on both checkpoints (held_out_ratios writes None when the error of predicting
    no change is 0: the stored balance token is one constant token per body over the whole ring at 380,550 and 415,075),
    and version 7's first paired NF judged a sense with no readable pair as not met, so NF missed at every checkpoint and
    stage 2 could never run. Now such a sense reads "not read", with its reason, beside the verdict and is not judged; NF
    is met on the senses it reads; a sense that does change is still judged; stage 2 runs where rule LS's look 1 is
    unresolved. Fails on ee2e66f10 (NF not met on the balance sense, stage 2 refused)."""
    import types
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    gains = [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER]
    row = _set_ratio(_pairs_row(start, gains, fatigue=f0), "vestibular", None, None)
    nf = M.rule_nf_pairs([row])
    ck = nf["checkpoints"][0]
    look = ck["looks"][0]
    assert nf["met"] and ck["state"] == "met" and look["met"], look
    bal = look["senses"]["vestibular"]
    assert bal["state"] == "not read" and bal["met"] is None and bal["pairs"] == 0 and bal["pairs_none_in_both_arms"] == 8
    assert "error of predicting no change is 0" in ck["not_read"]["vestibular"]
    assert ck["judged"] == ["tactile", "vision"] and ck["worse"] == {}
    rules = dict(LS=M.rule_ls_pairs([row], 3.0, 2.0, None, 0.069, 0.015, 0.117, 0.386), NF=nf)
    text = "\n".join(M.verdict_lines(rules))
    assert "rule NF version 7: met" in text and "balance: not read, not judged" in text
    assert "touch upper bound" in text and "vision upper bound" in text
    # a sense that changes is judged as before: the lesson's arms 3 in 100 worse in every pair miss
    worse = _set_ratio(_pairs_row(start, gains, fatigue=f0, r_other=1.03), "vestibular", None, None)
    got = M.rule_nf_pairs([worse])
    assert not got["met"] and got["checkpoints"][0]["state"] == "missed"
    assert set(got["checkpoints"][0]["worse"]) == {"vision"}, "a sense not read is never counted as worse"
    # stage 2 runs where look 1 left a target unresolved and the only sense NF cannot read is balance
    flip = [[g, 0.3 + j, 0.0, j, -j] for g, j in zip((0.3, -0.25, 0.28, -0.3, 0.1, -0.1, 0.2, -0.15), _JITTER)]
    unresolved = dict(_set_ratio(_pairs_row(start, flip, fatigue=f0), "vestibular", None, None), checkpoint="ckA",
                      EM=dict(met=True), loaded_step=400000)
    p = _stage1_doc(tmp_path, [unresolved])
    a = types.SimpleNamespace(line=3.0, deaf=2.0, skip="3", pairs=8, content_sigma_hi=0.069, fatigue_sigma_hi=0.015,
                              sigma_log=0.117, k6_fall_log=0.386)
    plan = M.stage1_plan(p, a)["plan"][0]
    assert plan["stage1_state"] == "unresolved" and plan["nf_met"] is True and plan["run"] is True, plan["why"]
    assert "balance not read" in plan["why"] and plan["nf_not_read"] == {"vestibular": ck["not_read"]["vestibular"]}


def test_nf_requires_every_pair_in_each_look_for_a_partly_read_sense(M):
    """Both looks require complete per-sense coverage. An entirely unread sense is
    excluded, but losing two readings at the second look must not produce a pass
    just because fourteen exceeds the first look's minimum of eight.
    """
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    gains = [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER]
    assert M.LINES["nf_min_pairs"] == 8 == M.LINES["ls_pairs"]
    six = _set_ratio(_pairs_row(start, gains, fatigue=f0), "audio", [1.0] * 6 + [None] * 2, [1.001] * 6 + [None] * 2)
    look = M.rule_nf_pairs([six])["checkpoints"][0]["looks"][0]
    assert look["senses"]["audio"]["state"] == "incomplete" and look["senses"]["audio"]["pairs"] == 6
    assert "read on 6 of 8 pairs, under the 8 the rule declares" in look["senses"]["audio"]["why"]
    assert not look["met"] and "audio" in look["judged"], "partial hearing evidence must prevent a no-harm pass"
    # Look 1 is complete. Look 2 loses two pairs and must refuse a pass.
    flip = [[g, 0.3 + j, 0.0, j, -j] for g, j in zip((0.3, -0.25, 0.28, -0.3, 0.1, -0.1, 0.2, -0.15), _JITTER)]
    both = _with_stage2(_pairs_row(start, flip, fatigue=f0), gains, fatigue=f0)
    c16 = [1.0] * 14 + [None] * 2
    r16 = [math.exp(d) if c is not None else None for c, d in zip(c16, [0.001, 0.002, 0.0, 0.001, 0.003, 0.0, 0, 0,
                                                                       0.002, 0.001, 0.0, 0.002, 0.001, 0.0, 0.003,
                                                                       0.001])]
    both = _set_ratio(both, "audio", c16, r16)
    nf = M.rule_nf_pairs([both])
    lk1, lk2 = nf["checkpoints"][0]["looks"]
    assert lk1["senses"]["audio"]["state"] == "met" and lk2["senses"]["audio"]["state"] == "incomplete"
    d = [math.log(x) for x in r16 if x is not None]
    st = M.pair_stats(d)
    assert lk2["senses"]["audio"]["pairs"] == 14 and lk2["senses"]["audio"]["upper_log"] == pytest.approx(st["hi"])
    assert not nf["met"] and "audio" in nf["checkpoints"][0]["judged"], "a complete first look cannot excuse an incomplete second look"
    # the lesson's arm lost the prediction the no-lesson arm made in one pair: missed, never "unread"
    lost = _set_ratio(_pairs_row(start, gains, fatigue=f0), "audio", 1.0, [1.0] * 7 + [float("nan")])
    look = M.rule_nf_pairs([lost])["checkpoints"][0]["looks"][0]
    assert look["senses"]["audio"]["state"] == "missed" and not look["met"]
    assert look["senses"]["audio"]["pairs_lost_in_the_lessons_arm"] == 1
    # no other sense read at all: nothing judged, never met
    blind = _pairs_row(start, gains, fatigue=f0)
    for p in blind["pairs"]:
        for arm in ("C", "R"):
            for x in p[arm]["rows"]:
                x["ratios"] = {"proprioception": 1.1, "vestibular": None}
    got = M.rule_nf_pairs([blind])
    assert not got["met"] and got["checkpoints"][0]["state"] == "nothing judged"
    assert got["checkpoints"][0]["looks"][0]["state"] == "nothing judged"


def test_rule_ls_reports_the_holds_share_and_the_attributions_bound_beside_the_gains(M):
    """The review's findings 3 and 5, readings that decide nothing: per body, the share of arm R's updates in which the
    hold acted and in which the push did (per pair, from the lesson's own gauges), their association with the gains over
    the pairs, beside the no-lesson wander's; the lesson's attribution over the measured gain per pair and its BOUND, the
    mean attribution over the gain's lower bound (at least 1), which the watcher's A1 (ii) divides by."""
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    gains = [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER]
    held = [[40 * (s + 1), 400, 0, 0, 0] for s in range(8)]
    pushed = [[300, 300, 0, 0, 0] for _s in range(8)]
    att = [[0.45, 0.4, 0.0, 0.01, 0.0] for _s in range(8)]
    row = _pairs_row(start, gains, fatigue=f0, held=held, pushed=pushed, attributed=att)
    got = M.rule_ls_pairs([row], 3.0, 2.0, None, 0.069, 0.015, 0.117, 0.386)
    b1 = got["checkpoints"][0]["bodies"][0]
    assert got["state"] == "met", "the readings decide nothing"
    assert b1["hold"]["held_share"] == pytest.approx([40 * (s + 1) / 400 for s in range(8)])
    assert b1["hold"]["pushed_share_mean"] == pytest.approx(0.75)
    g = [0.3 + j for j in _JITTER]
    hs = [40 * (s + 1) / 400 for s in range(8)]
    mg, mh = sum(g) / 8, sum(hs) / 8
    r = sum((a - mg) * (c - mh) for a, c in zip(g, hs)) / math.sqrt(sum((a - mg) ** 2 for a in g)
                                                                   * sum((c - mh) ** 2 for c in hs))
    assert b1["hold"]["gain_with_held_share"] == pytest.approx(r)
    assert got["checkpoints"][0]["bodies"][1]["hold"]["gain_with_held_share"] is None, "no spread: no association"
    assert b1["attributed_mean"] == pytest.approx(0.45)
    assert b1["over_read_bound"] == pytest.approx(max(1.0, 0.45 / b1["gain_lower"]))
    assert b1["attributed_over_measured"]["high"] == pytest.approx(max(0.45 / x for x in g))
    b4 = got["checkpoints"][0]["bodies"][3]
    assert b4["over_read_bound"] is None, "a body whose lower bound is not over 0: unbounded"


def test_version_7_runs_its_stage_1_pairs_on_seeds_13_to_20_on_one_set_of_clip_factors(M, monkeypatch):
    """measure_checkpoint with pairs K (record section 47 and its fix round): the checkpoint's clip factors are drawn
    once and every arm gets them; stage 1's pair s runs arm C and arm R on the seed 13 + s - 1 (none version 6 drew:
    the review's BLOCKER was a judged pair on seed 11, version 6's arms replayed), arm R reading its gain against its own
    pair's arm C; the held-out ratios at update 0 and at the last update; the row is written as each pair ends. Stage 2
    carries stage 1's factors, clip record, EM, T and pairs and runs seeds 21-28 on the same factors, loading no EM or
    T again. Without pairs, version 6's three arms. Fails on 67335783f (seeds 11-18, no stages)."""
    import types
    calls, drawn, saved = [], [], []
    model = types.SimpleNamespace(forward_model=types.SimpleNamespace(eval=lambda: None))
    em = []
    monkeypatch.setattr(M, "em_reading", lambda ck: em.append(ck) or dict(met=True))
    monkeypatch.setattr(M, "load_model", lambda ck, repair=False, **kw: types.SimpleNamespace(
        model=model, load=dict(step=415075), info=dict(checkpoint=ck)))
    monkeypatch.setattr(M, "load_ring", lambda frozen: ({}, None, "note"))
    monkeypatch.setattr(M, "Ring", lambda rep, nf: types.SimpleNamespace(B=8, N=6000))
    monkeypatch.setattr(M, "clip_factor", lambda ck: dict(factor=0.005, median_norm=200.0, records=20))
    factors = [0.1, 0.2]

    def shared(model_, ring, ck, seed=None):
        drawn.append(ck)
        return factors
    monkeypatch.setattr(M, "shared_clip_factors", shared)

    def arm_(model_, ring, ck, arm, updates, seed, save, rows=None, reference=None, factors=None, heldout_at=None,
             **kw):
        calls.append(dict(arm=arm, seed=seed, reference=reference, factors=factors, heldout_at=heldout_at))
        return dict(arm=arm, seed=seed, rows=[dict(update=0, arm=arm, seed=seed), dict(update=updates)])
    monkeypatch.setattr(M, "refit_arm", arm_)

    def save(row=None):
        saved.append(len((row or {}).get("pairs") or []))
    row = M.measure_checkpoint("ck", 400, save, line=3.75, deaf=2.86, skip="3", pairs=8)
    assert drawn == ["ck"] and em == ["ck"], "one set of clip factors per checkpoint; EM on stage 1"
    assert row["pairs_declared"] == 8 and [p["seed"] for p in row["pairs"]] == list(range(13, 21))
    assert all(p["stage"] == 1 for p in row["pairs"]) and row["stage"] == 1
    assert row["clip_factors_shared"] is factors and row["clip_record"]["records"] == 20
    assert [(c["arm"], c["seed"]) for c in calls] == [(a, s) for s in range(13, 21) for a in ("C", "R")]
    assert all(c["factors"] is factors and c["heldout_at"] == {0, 400} for c in calls)
    for p in row["pairs"]:
        rc = [c for c in calls if c["arm"] == "R" and c["seed"] == p["seed"]][0]
        assert rc["reference"] is p["C"]["rows"], "arm R reads its gain against its own pair's arm C"
    assert saved.count(8) >= 2, "the row is written again as the last pair ends"
    # stage 2 on stage 1's row: seeds 21-28 on stage 1's factors, no EM, no new factors
    calls.clear()
    stage1 = dict(row, loaded_step=415075, pairs=[dict(p, C=dict(p["C"], rows=[dict(update=0), dict(update=400)]),
                                                       R=dict(p["R"], rows=[dict(update=0), dict(update=400)]))
                                                  for p in row["pairs"]])
    two = M.measure_checkpoint("ck", 400, save, line=3.75, deaf=2.86, skip="3", pairs=8, stage=2, stage1_row=stage1)
    assert drawn == ["ck"] and em == ["ck"], "stage 2 draws no factors and reads no EM"
    assert [p["seed"] for p in two["pairs"]] == list(range(13, 21)) + list(range(21, 29))
    assert [p["stage"] for p in two["pairs"]] == [1] * 8 + [2] * 8 and two["EM_from"].startswith("stage 1")
    assert all(c["factors"] == factors for c in calls) and [c["seed"] for c in calls if c["arm"] == "C"] == \
        list(range(21, 29))
    with pytest.raises(ValueError, match="8 and 8"):
        M.measure_checkpoint("ck", 400, save, line=3.75, deaf=2.86, pairs=5, stage=2, stage1_row=stage1)
    with pytest.raises(ValueError, match="ended at updates"):
        M.measure_checkpoint("ck", 200, save, line=3.75, deaf=2.86, pairs=8, stage=2, stage1_row=stage1)
    calls.clear()
    old = M.measure_checkpoint("ck", 400, save, line=3.75, deaf=2.86, skip="3")
    assert "pairs" not in old and set(old["arms"]) == {"C", "R", "C2"}
    assert [(c["arm"], c["seed"], c["factors"], c["heldout_at"]) for c in calls] == [
        ("C", 11, None, None), ("R", 11, None, None), ("C2", 12, None, None)], "version 6's arms, unchanged"
    with pytest.raises(ValueError, match="two pairs"):
        M.measure_checkpoint("ck", 400, save, line=3.75, deaf=2.86, pairs=1)


def _stage1_doc(tmp_path, rows, name="stage1", **over):
    doc = dict(command="measure", ls_version=7, stage=1, finished="2026-09-27 20:00:00", line=3.0, deaf=2.0, skip=[3],
               pairs=8, rows=rows)
    doc.update(over)
    p = tmp_path / name / "proprio_token_measure.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_stage_2_runs_only_where_stage_1_left_a_checkpoint_unresolved(M, tmp_path, monkeypatch):
    """The coordinator's decision: stage 2's jobs run only for bodies stage 1 leaves unresolved. measure --stage 2 reads
    the stage-1 result, re-judges it, and runs stage 2 only at a checkpoint whose look 1 read "unresolved" with NF met
    and EM met; with none it writes "not needed" and loads no model. A stage-1 result of other settings, or one not
    finished, is refused; a checkpoint stage 1 did not read is refused. Fails on 67335783f (no --stage)."""
    import types
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    flip = [[g, 0.3 + j, 0.0, j, -j] for g, j in zip((0.3, -0.25, 0.28, -0.3, 0.1, -0.1, 0.2, -0.15), _JITTER)]
    good = [[0.3 + j, 0.3 - j, 0.0, j, -j] for j in _JITTER]
    unresolved = dict(_pairs_row(start, flip, fatigue=f0), checkpoint="ckA", EM=dict(met=True), loaded_step=400000)
    met = dict(_pairs_row(start, good, fatigue=f0), checkpoint="ckB", EM=dict(met=True), loaded_step=401000)
    seen = []

    def fake(ck, updates, save, **kw):
        seen.append((ck, kw.get("stage"), (kw.get("stage1_row") or {}).get("checkpoint")))
        return dict(checkpoint=ck, loaded_step=400000)
    monkeypatch.setattr(M, "measure_checkpoint", fake)

    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(dict(rules=dict(version=6, LINE=3.0, DEAF=2.0, SKIP=[3],
                                              window_noise=dict(sigma=0.117, sigma_hi=0.171, k6_fall=0.386),
                                              content_noise=dict(sigma=0.05, sigma_hi=0.069),
                                              fatigue_noise=dict(sigma=0.01, sigma_hi=0.015)))), encoding="utf-8")

    def args(path, out, **kw):
        base = dict(cmd="measure", output=str(tmp_path / out), updates=400, refit_steps=1, checkpoints=[], line=None,
                    lam=None, strength=None, steps=1, skip=None, pairs=8, stage=2, stage1_result=str(path),
                    from_calibration=str(cal))
        base.update(kw)
        return types.SimpleNamespace(**base)
    p = _stage1_doc(tmp_path, [unresolved, met])
    assert M.run(args(p, "s2")) == 0
    assert seen == [("ckA", 2, "ckA")], "stage 2 at the unresolved checkpoint only"
    doc = json.loads((tmp_path / "s2" / "proprio_token_measure.json").read_text(encoding="utf-8"))
    plan = {x["step"]: x for x in doc["stage2_plan"]}
    assert plan[400000]["run"] and not plan[401000]["run"] and "cannot change the outcome" in plan[401000]["why"]
    seen.clear()
    p2 = _stage1_doc(tmp_path, [met], name="stage1b")
    assert M.run(args(p2, "s2b")) == 0 and seen == [], "nothing unresolved: no model is loaded"
    doc = json.loads((tmp_path / "s2b" / "proprio_token_measure.json").read_text(encoding="utf-8"))
    assert "stage 2 not needed" in doc["rules"]["note"]
    with pytest.raises(SystemExit, match="line"):
        M.run(args(_stage1_doc(tmp_path, [unresolved], name="c", line=3.5), "s2c"))
    with pytest.raises(SystemExit, match="not a finished"):
        M.run(args(_stage1_doc(tmp_path, [unresolved], name="d", finished=None), "s2d"))
    with pytest.raises(SystemExit, match="did not read"):
        M.run(args(p, "s2e", checkpoints=["ckZ"]))


def test_stage_2_does_not_run_where_stage_1s_nf_missed(M, tmp_path):
    """The re-review's finding 6 (a planted defect no test caught: stage 2 run where NF missed; the stage-2 test above
    builds NF-met rows only): a checkpoint whose look 1 left a target unresolved but whose NF missed (the lesson's arms
    read vision 3 in 100 worse in every pair) gets no stage 2, since NF's miss already fails it and 16 more arms could
    not change the outcome; beside it a checkpoint with the same unresolved look 1 and NF met runs stage 2."""
    import types
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    f0 = [0.10, 0.09, 0.05, 0.15, 0.12]
    flip = [[g, 0.3 + j, 0.0, j, -j] for g, j in zip((0.3, -0.25, 0.28, -0.3, 0.1, -0.1, 0.2, -0.15), _JITTER)]
    nf_missed = dict(_pairs_row(start, flip, fatigue=f0, r_other=1.03), checkpoint="ckA", EM=dict(met=True),
                     loaded_step=400000)
    nf_met = dict(_pairs_row(start, flip, fatigue=f0), checkpoint="ckB", EM=dict(met=True), loaded_step=401000)
    p = _stage1_doc(tmp_path, [nf_missed, nf_met])
    a = types.SimpleNamespace(line=3.0, deaf=2.0, skip="3", pairs=8, content_sigma_hi=0.069, fatigue_sigma_hi=0.015,
                              sigma_log=0.117, k6_fall_log=0.386)
    plan = {x["step"]: x for x in M.stage1_plan(p, a)["plan"]}
    assert plan[400000]["stage1_state"] == "unresolved" and plan[400000]["nf_met"] is False
    assert plan[400000]["run"] is False and "NF missed" in plan[400000]["why"], plan[400000]["why"]
    assert plan[401000]["stage1_state"] == "unresolved" and plan[401000]["run"] is True


def test_a_job_ended_early_keeps_its_finished_pairs_and_a_rerun_keeps_the_earlier_file(M, tmp_path, monkeypatch):
    """The coordinator's decision E: the result file is written whole (through a temporary file and one rename) as each
    pair ends, so a job the pause watchdog ends keeps every pair it finished; a job queued again moves the earlier file
    aside under its own time instead of overwriting it; the verdict still needs every declared pair. Fails on 67335783f
    (a plain overwrite, and a rerun replaced the earlier file)."""
    import types
    start = [1.0, 1.2, 0.3, 4.0, 3.9]
    two = _pairs_row(start, [[0.3, 0.3, 0.0, 0.0, 0.0]] * 2, declared=8)

    def fake(ck, updates, save, **kw):
        row = dict(checkpoint=ck, loaded_step=400000, pairs_declared=8, pairs=[])
        save(row)
        for p in two["pairs"]:
            row["pairs"].append(p)
            save(row)
        raise KeyboardInterrupt                                   # the watchdog ends the tree mid-run
    monkeypatch.setattr(M, "measure_checkpoint", fake)
    a = types.SimpleNamespace(cmd="measure", output=str(tmp_path), updates=400, refit_steps=1, checkpoints=["ck1"],
                              line=3.0, strength=None, steps=12, pairs=8)
    with pytest.raises(KeyboardInterrupt):
        M.run(a)
    path = tmp_path / "proprio_token_measure.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["rows"][0]["pairs"]) == 2 and not (tmp_path / "proprio_token_measure.json.tmp").exists()
    assert M.rule_ls_pairs(doc["rows"], 3.0, 2.0)["state"] == "incomplete", "no verdict on 2 of 8 pairs"
    with pytest.raises(KeyboardInterrupt):
        M.run(a)
    kept = sorted(tmp_path.glob("proprio_token_measure.earlier_*.json"))
    assert len(kept) == 1 and len(json.loads(kept[0].read_text(encoding="utf-8"))["rows"][0]["pairs"]) == 2
    assert json.loads(path.read_text(encoding="utf-8"))["earlier_result_kept_at"] == str(kept[0])


def test_a_write_cut_off_half_way_leaves_the_last_whole_result_file(M, tmp_path, monkeypatch):
    """The re-review's finding 6 (a planted defect no test caught: a plain, non-atomic write; the review judged it not
    testable in one process). A job the pause watchdog ends while it writes must leave the last whole file. Here a write
    is cut off half way (half the text written, then the job ended): through write_atomic the result file still holds
    the last whole text, where a plain write leaves half a file no JSON reader opens. Then run()'s own saves: a measure
    job cut off while writing its second pair keeps the first pair's whole file."""
    import types
    real = Path.write_text
    state = dict(n=0, cut=None)

    def cut_write(self, data, *a, **k):
        state["n"] += 1
        if state["cut"] is not None and state["n"] == state["cut"]:
            real(self, data[: len(data) // 2], *a, **k)
            raise KeyboardInterrupt("the pause watchdog ends the job mid-write")
        return real(self, data, *a, **k)
    monkeypatch.setattr(Path, "write_text", cut_write)
    path = tmp_path / "proprio_token_measure.json"
    first = dict(rows=[dict(pairs=[1, 2])])
    M.write_atomic(path, json.dumps(first))
    state["cut"] = state["n"] + 1
    with pytest.raises(KeyboardInterrupt):
        M.write_atomic(path, json.dumps(dict(rows=[dict(pairs=[1, 2, 3])])))
    assert json.loads(path.read_text(encoding="utf-8")) == first, "the last whole file stays"
    two = _pairs_row([1.0, 1.2, 0.3, 4.0, 3.9], [[0.3, 0.3, 0.0, 0.0, 0.0]] * 2, declared=8)

    def fake(ck, updates, save, **kw):
        row = dict(checkpoint=ck, loaded_step=400000, pairs_declared=8, pairs=[])
        for p in two["pairs"]:
            row["pairs"].append(p)
            save(row)
            state["cut"] = state["n"] + 1                 # the next write, the second pair's, is cut half way
        return row
    monkeypatch.setattr(M, "measure_checkpoint", fake)
    state["cut"] = None
    out = tmp_path / "job"
    a = types.SimpleNamespace(cmd="measure", output=str(out), updates=400, refit_steps=1, checkpoints=["ck1"],
                              line=3.0, strength=None, steps=12, pairs=8)
    with pytest.raises(KeyboardInterrupt):
        M.run(a)
    doc = json.loads((out / "proprio_token_measure.json").read_text(encoding="utf-8"))
    assert len(doc["rows"][0]["pairs"]) == 1, "the first pair's whole write stands"


def _graded(step, terms_by_body, ring=None):
    """A graded checkpoint: terms_by_body[b] = {term: effect per step} for 4 steps (halves equal)."""
    bodies = [dict(body=b + 1, steps=[dict(terms=dict(t), total=sum(t.values()), unattributed=0.0)
                                      for _ in range(4)]) for b, t in enumerate(terms_by_body)]
    return dict(loaded_step=step, live=dict(bodies=bodies), ring=ring or {})


def test_rule_p_names_the_pressure_whose_change_accounts_for_the_net(M):
    before = {"next_frame.proprioception": -1e-6, "tenure_tok": 1e-6}
    after = {"next_frame.proprioception": -1e-6, "tenure_tok": 1e-6, "efference_body": -8e-6, "contact_head": -1e-6}
    rows = [_graded(s, [before] * 8) for s in (288771, 332501, 344400)]
    rows += [_graded(s, [after] * 8) for s in (350926, 363041, 415000)]
    out = M.rule_p(rows)
    assert out["enough"] and out["named"] == "efference_body"
    assert out["terms"]["efference_body"]["delta"] == pytest.approx(-8e-6)
    assert "efference_body" in out["eroding"] and not out["supporting"]
    split = rows[:3] + [_graded(s, [dict(after, contact_head=-8e-6)] * 8) for s in (350926, 363041, 415000)]
    assert M.rule_p(split)["named"] in ("efference_body", "contact_head")
    three = rows[:3] + [_graded(s, [dict(after, contact_head=-8e-6, tenure=-8e-6)] * 8)
                        for s in (350926, 363041, 415000)]
    assert M.rule_p(three)["named"] == "no single pressure", "no term carries half of the net"
    only3 = rows[:3] + [_graded(s, [before, before, dict(before, efference_body=-9e-5)] + [before] * 5)
                        for s in (350926, 363041, 415000)]
    assert M.rule_p(only3)["named"] == "no single pressure"
    assert M.rule_p(rows[:2] + rows[3:])["named"] == "no single pressure"


def test_the_heading_readings_leave_the_model_as_it_was_and_the_refit_closes_on_todays_tokens(M, monkeypatch):
    monkeypatch.setitem(M.LINES, "eval_windows", 2)
    m = _lite()
    _prime_norm(m)
    rep, nf_raw = _stand_in_ring(m)
    ring = M.Ring(rep, nf_raw)
    before = [p.detach().clone() for p in m.params]
    out = M.heading_readings(m, ring, refit_steps=60)
    assert all(torch.equal(a, b) for a, b in zip(before, m.params)), "the model is not changed"
    assert set(out["ratios"]) == {"as_is", "switch", "switch_after_refit"}
    hc = out["head_change_over_own_change"]
    assert set(hc) == {"switch", "switch_after_refit"} and "proprioception" in hc["switch"]
    assert hc["switch"]["proprioception"] > 0.0
    assert hc["switch_after_refit"]["proprioception"] < hc["switch"]["proprioception"]
