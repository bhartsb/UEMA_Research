# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The pause measurement of the vestibular encoder repair (docs/gain_of_function_v4/
vestibular_encoder_measurements.py), on the lite model with production's constant put into its vestibular
encoder: no checkpoint and no companion, so this runs outside a training pause.

What is held here: the script runs this tree's code and says so; the delayed lesson's forward it builds reads
the vestibular token from the saved raw input and every other sense from the saved tokens, in float32 and bf16,
and can put a revived token (or one per past moment) into the context and into the memory's events; its comparison
finds the real switch within the switch's line on the token, the core and every head in both precisions; in the
CPU's emulation of bf16 the dead token is one value in every body and the anti-collapse lesson moves nothing
(review 2, finding 4), while float32 and the float32-encoder arm revive it; the learning arms put the model back
after each arm, precision included; the learning context is built from the arm's own snapshots; the reach over
several saved moments gives the declared cost from the context forms; the lineage's earlier checkpoints are found
in the run folders and the archive; the transient with the lesson ceiling runs; the rules (MODE in the run's
precision) are applied as declared. The third review's fix round: EM32 is read whatever the mode and the float32
mode is never chosen with it unread or past its line; RV and the cost come from the arm of the mode chosen; the
result is written after every stage and every arm and a stopped job keeps it; the CPU fallback does nothing when
the card's result holds the cost; OT reads the collapse watch's own statistic on clusters of saved moments and
gates the mode at K2's line (the watcher's); T and the CORE arm read the lesson weight the model trains with; the
rehearsal's share line fails a report with no share. The fourth review's fix round: RV and OT are read on both deploy
arms, and a mode whose arm holds no OT (or one read in another precision) or no declared cost is no mode (the review's
planted case is refused); a card that read the pause's own checkpoint to its end leaves the CPU fallback nothing to do.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mimo_bridge import vestibular_sense as VS
from mimo_bridge.lem_model import LemLite

HERE = Path(__file__).resolve()
SCRIPT = HERE.parents[4] / "docs" / "gain_of_function_v4" / "vestibular_encoder_measurements.py"
# the deploy arm of each mode (the script's RV_ARMS, inverted), written out so a planted row does not lean on the code
# under test
ARM_MODE = {"CORE_bf16": "rescale_restart", "CORE32_bf16": "rescale_restart_fp32"}
# a declared cost the kill-line watcher accepts holds a rise for each of vision, touch, hearing and body position
RISES = {"audio": 0.005, "vision": 0.001, "tactile": 0.001, "proprioception": 0.001}


def _script():
    spec = importlib.util.spec_from_file_location("vestibular_encoder_measurements", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dead_lite(seed=3, **kw):
    torch.set_num_threads(1)
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                top_down_gate=True, body_layout="trunk_v1", vestibular_sense=True, **kw)
    last = m.encoder_registry.encoders["vestibular"].mlp[-1]
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        v = torch.randn(last.bias.shape, generator=g, dtype=torch.float64)
        last.bias.copy_(((v - v.mean()) / v.std(unbiased=False) * 74425.156176 - 511.889836).float())
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.0097)
    for p in m.params:
        m.optimizer.state[p] = {"step": torch.tensor(66725.0),
                                "exp_avg": torch.randn(p.shape, generator=g) * 4.3e-11,
                                "exp_avg_sq": (torch.rand(p.shape, generator=g) * 4.3e-10) ** 2}
    return m


def _moment(m, B=8, seed=11):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, VS.ENCODER_DIM, generator=g).clamp(-4, 4)
    toks = {mm: torch.nn.functional.layer_norm(torch.randn(B, 1, m.d_model, generator=g), (m.d_model,))
            for mm in m.active}
    mask = {mm: "stale" for mm in m.active}
    mask["vestibular"] = "real"
    ctx = torch.randn(B, (m.ctx_len - 1) * m.n_mod, m.d_model, generator=g)
    return dict(_nf_raw={"vestibular": z}, _nf_mask=mask, _nf_ctx=ctx, _nf_tokens=toks, _nf_memory=None)


def _target(m, nf):
    """The moment's own tokens (last_tokens), every graded sense real."""
    return {mm: v.clone() for mm, v in nf["_nf_tokens"].items()}


def _with_averages(m, B=8):
    """The saved averages of ckpt_step_411150 in shape: the vestibular ones dead, the others alive."""
    m.nf_norm.load_state_dict({"stats": {"vestibular": [1.1e-13, 1.0, 66723], "audio": [0.0465, 1.0, 66724],
                                          "tactile": [0.017, 1.0, 66724], "vision": [0.0061, 1.0, 66724],
                                          "proprioception": [0.0025, 1.0, 66724]}})
    st = {}
    for key, (mf, pf, ms, ps) in dict(vestibular=(1.15e-5, 1.2e-20, 5.0e-6, 1.1e-13),
                                      audio=(0.05, 0.05, 0.06, 0.05), tactile=(0.015, 0.017, 0.016, 0.017)).items():
        st[key] = {"mf": torch.full((B,), mf), "pf": torch.full((B,), pf), "wf": torch.ones(B),
                   "ms": torch.full((B,), ms), "ps": torch.full((B,), ps), "ws": torch.ones(B)}
    m.wellness_channels.lp.load_state_dict({"state": st, "holdout": {}})
    return m


def test_the_script_runs_this_trees_code():
    S = _script()
    where = S.assert_this_tree(S.VS)
    assert Path(next(iter(where.values()))).is_relative_to(HERE.parents[2])
    assert S.SRC == HERE.parents[2]
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match="not from this worktree"):
        S.assert_this_tree(SimpleNamespace(__name__="elsewhere", __file__="C:/somewhere/else/mod.py"))


def test_the_forward_reads_the_vestibular_input_and_the_saved_tokens():
    S = _script()
    m = _dead_lite()
    nf = _moment(m)
    f = S.lesson_forward(m, nf)
    want = m.encoder_registry.encode({"vestibular": nf["_nf_raw"]["vestibular"]},
                                     {"vestibular": "real"})["vestibular"].squeeze(1)
    assert torch.equal(f["vest"], want)
    assert set(f["preds"]) == set(m.pred_mods)
    other = torch.nn.functional.layer_norm(torch.randn(8, m.d_model), (m.d_model,))
    newest = S.lesson_forward(m, nf, vest_token=other)
    both = S.lesson_forward(m, nf, vest_token=other, context_too=True)
    assert not torch.equal(newest["last"], f["last"]) and not torch.equal(both["last"], newest["last"])
    k = m.active.index("vestibular")
    T = nf["_nf_ctx"].shape[1] // m.n_mod
    emb = m.forward_model.modality_embedding.weight[k].detach()
    # one token in every past moment, or one per past moment (oldest first); nothing else moves
    for tok in (other, torch.randn(8, T, m.d_model)):
        c = S._context_with(m, nf["_nf_ctx"], tok).reshape(8, T, m.n_mod, m.d_model)
        c0 = nf["_nf_ctx"].reshape(8, T, m.n_mod, m.d_model)
        want_k = (tok[:, None] if tok.dim() == 2 else tok) + emb
        assert torch.allclose(c[:, :, k], want_k.expand_as(c[:, :, k]))
        keep = [i for i in range(m.n_mod) if i != k]
        assert torch.equal(c[:, :, keep], c0[:, :, keep])
    fb = S.lesson_forward(m, nf, precision="bfloat16")
    assert fb["last"].dtype == torch.float32 and not torch.equal(fb["last"], f["last"])
    assert float((fb["last"] - f["last"]).abs().max()) < 0.05 * float(f["last"].abs().max())


def test_the_real_switch_is_within_its_line_on_every_output_in_both_precisions():
    S = _script()
    m = _dead_lite()
    nf = _moment(m)
    fa = {p: S.lesson_forward(m, nf, precision=p) for p in S.EM_PRECISIONS}
    rec = VS.repair_encoder(m, load_optimizer=True, restart_moments=True)
    fr = {p: S.lesson_forward(m, nf, precision=p) for p in S.EM_PRECISIONS}
    assert rec["switched"] and rec["divided_by"] == 4096.0
    for p in S.EM_PRECISIONS:
        em = S.compare(fa[p], fr[p])
        assert all(v["within"] for v in em.values()), (p, {k: v for k, v in em.items() if not v["within"]})
    assert torch.equal(fa["bfloat16"]["last"], fr["bfloat16"]["last"])
    fr["float32"]["last"] = fr["float32"]["last"] + 1e-3
    assert not S.compare(fa["float32"], fr["float32"])["core_newest_step"]["within"]


def test_in_bf16_the_dead_token_is_one_value_and_the_anti_collapse_lesson_moves_nothing():
    # review 2, finding 4, on the lite model: the weights' part after the switch is a ten-millionth of the bias,
    # below bf16's step; float32 and the float32-encoder arm keep it
    S = _script()
    m = _dead_lite()
    z = _moment(m)["_nf_raw"]["vestibular"]
    VS.repair_encoder(m, load_optimizer=True, restart_moments=True)
    t16 = S.token_of(m, z, "bfloat16")
    t32 = S.token_of(m, z, "float32")
    assert S.numbers_varying(t16) == 0 and S.numbers_varying(t32) > 0
    o16 = S.learn(m, z, 20, "bfloat16")
    assert o16["readings"]["20"]["spread"] == 0.0 and o16["readings"]["20"]["numbers_varying"] == 0
    VS.apply_encoder_precision(m, "rescale_restart_fp32")
    t_fp32_encoder = S.token_of(m, z, "bfloat16")          # the run's autocast, this encoder in float32
    assert S.numbers_varying(t_fp32_encoder) > 0
    assert torch.allclose(t_fp32_encoder.double(), t32.double(), atol=1e-6)


def test_the_learning_arms_do_not_leak_and_read_each_precision():
    S = _script()
    m = _dead_lite()
    nf = _moment(m)
    z = nf["_nf_raw"]["vestibular"]
    w0, m0 = S.encoder_state(m)
    out = S.ls_arms(m, z, 100, nf=nf, target=_target(m, nf))
    w1, m1 = S.encoder_state(m)
    for n in w0:                                      # every arm put the model back
        assert torch.equal(w0[n], w1[n]), n
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(m0[n][k], m1[n][k]), (n, k)
    enc = m.encoder_registry.encoders["vestibular"]
    assert "forward" not in enc.__dict__ and not getattr(enc, "vestibular_repair_fp32", False)
    arms = out["arms"]
    assert set(arms) == {a[0] for a in S.ARMS} and all("error" not in v for v in arms.values()), \
        {k: v.get("error") for k, v in arms.items() if "error" in v}
    a = arms["A"]["readings"]["100"]["spread"]
    for arm in ("R", "RR"):
        r = arms[arm]["readings"]["100"]["spread"]
        assert a < 1e-4 and r > 0.05 and r > 1000 * a, (arm, a, r)
    assert arms["A_bf16"]["readings"]["100"]["spread"] == 0.0
    assert arms["RR_bf16"]["readings"]["100"]["spread"] == 0.0
    assert arms["CORE32_bf16"]["readings"]["100"]["numbers_varying"] > 0
    assert arms["R"]["switch"]["divided_by"] == 4096.0 and arms["R8"]["switch"]["divided_by"] == 512.0
    assert VS.repair_divisor(74425.156176, 1e-5) == 4096.0
    # the revived arms kept for RV and OT are the ones a deploy runs (third review, section E): the whole lesson
    # through the core in bf16, and with this encoder in float32; each token in the precision the deploy computes it
    assert set(out["revived"]) == {"CORE_bf16", "CORE32_bf16"}
    rv = out["revived"]["CORE_bf16"]
    assert rv["token"].shape == (8, m.d_model) and max(rv["snapshots"]) == 100 and len(rv["snapshots"]) == 13
    assert rv["precision"] == "bfloat16" and out["revived"]["CORE32_bf16"]["precision"] == "float32"


def test_the_learning_context_is_built_from_the_arms_snapshots():
    S = _script()
    m = _dead_lite()
    nf = _moment(m)
    z = nf["_nf_raw"]["vestibular"]
    VS.repair_encoder(m, load_optimizer=True, restart_moments=True)
    o = S.learn(m, z, 40, snapshots=True)
    T = nf["_nf_ctx"].shape[1] // m.n_mod
    ctx = S.learning_context(m, z, dict(token=o["final_token"], snapshots=o["snapshots"], updates=40), T)
    assert ctx.shape == (8, T, m.d_model)
    # the newest past moment is the final encoder's token; the encoder is put back as it was
    assert torch.allclose(ctx[:, -1], S.token_of(m, z), atol=1e-6)
    assert S.learning_context(m, z, dict(token=o["final_token"], snapshots={}, updates=40), T) is None


def _memory_snapshot(m, B=8, K=6, seed=13):
    g = torch.Generator().manual_seed(seed)
    times = torch.arange(K, dtype=torch.float32).repeat(B, 1) * 0.5
    return dict(events=0.3 * torch.randn(B, K, m.persistent_memory.input_size, generator=g), times=times,
                now=times[:, -1].clone(), carry=None, carry_time=None)


def test_the_reach_reads_every_form_and_the_memory_path():
    S = _script()
    m = _dead_lite(persistent_memory="on")
    nf = _moment(m)
    nf["_nf_memory"] = _memory_snapshot(m)
    revived = torch.nn.functional.layer_norm(torch.randn(8, m.d_model), (m.d_model,))
    mem = S._memory_with(m, nf["_nf_memory"], revived)
    k = m.active.index("vestibular")
    d = m.d_model
    ev0, ev1 = nf["_nf_memory"]["events"], mem["events"]
    assert torch.equal(ev1[:, :, k * d:(k + 1) * d], revived[:, None].expand(8, ev1.shape[1], d))
    assert torch.equal(torch.cat([ev1[:, :, :k * d], ev1[:, :, (k + 1) * d:]], -1),
                       torch.cat([ev0[:, :, :k * d], ev0[:, :, (k + 1) * d:]], -1))
    with torch.no_grad():
        torch.nn.init.normal_(m.persistent_memory.readout.weight, std=0.05, generator=torch.Generator().manual_seed(4))
    fa = S.lesson_forward(m, nf)
    ctx_only = S.lesson_forward(m, nf, vest_token=revived, context_too=True)
    with_mem = S.lesson_forward(m, nf, vest_token=revived, context_too=True, memory_too=True)
    assert not torch.equal(with_mem["last"], ctx_only["last"])
    T = nf["_nf_ctx"].shape[1] // m.n_mod
    learning = torch.nn.functional.layer_norm(torch.randn(8, T, m.d_model), (m.d_model,))
    rv = S.reach(m, nf, fa, revived, learning)
    assert set(rv) == {"newest", "with_context", "learning", "with_memory", "revived_spread_between_bodies"}
    assert all(set(rv[f]) == set(m.pred_mods) - {"vestibular"} for f in ("newest", "with_context", "learning",
                                                                          "with_memory"))
    nf["_nf_memory"] = None
    rv = S.reach(m, nf, S.lesson_forward(m, nf), revived)
    assert rv["with_memory"] is None and rv["learning"] is None


def test_the_declared_cost_comes_from_the_context_forms_over_the_moments():
    S = _script()

    def rv(newest, ctx, learning, memory):
        f = lambda v: {"audio": {"expected_ratio_rise": v}}      # noqa: E731
        return dict(newest=f(newest), with_context=f(ctx), learning=None if learning is None else f(learning),
                    with_memory=f(memory))
    # review 2's readings at 411,150: newest 0.315, context and memory 0.0039; at 411,125 newest 0.051, context 0.0053
    got = S.rises_over_moments([rv(0.315, 0.0039, 0.004, 0.0039), rv(0.051, 0.0053, None, 0.0053)])
    assert got["first_report"]["audio"] == pytest.approx((0.315 + 0.051) / 2)
    assert got["later_reports"]["audio"] == pytest.approx((0.004 + 0.0053) / 2)


def test_the_lineage_checkpoints_before_a_checkpoint_are_found_in_runs_and_the_archive(tmp_path):
    S = _script()
    exp, arc = tmp_path / "experiments", tmp_path / "archive"
    parent, run = exp / "H84_seg40", exp / "H84_seg40_resume_1"
    for p in (parent, run):
        (p / "checkpoints").mkdir(parents=True)
    (parent / "args.json").write_text("{}", encoding="utf-8")
    (run / "args.json").write_text(json.dumps({"resume": str(parent / "checkpoints" / "ckpt_step_411125")}),
                                   encoding="utf-8")
    for p in (run / "checkpoints" / "ckpt_step_411150", run / "checkpoints" / "ckpt_step_411300",
              parent / "checkpoints" / "ckpt_step_411125", arc / "H84_seg40" / "checkpoints" / "ckpt_step_410400",
              arc / "H84_seg40" / "checkpoints" / "ckpt_step_408450"):
        p.mkdir(parents=True)
    got = S.lineage_checkpoints(run / "checkpoints" / "ckpt_step_411300", 3, archive=arc)
    assert [p.name for p in got] == ["ckpt_step_411150", "ckpt_step_411125", "ckpt_step_410400"]


def test_the_transient_with_the_lesson_ceiling():
    S = _script()
    m = _with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True))
    T = S.transients(m, P_values=(1e-8, 2.5e-3), multiples=(11.7,), updates=600)
    slow = next(r for r in T["rows"] if r["P_start"] == 1e-8)
    fast = next(r for r in T["rows"] if r["P_start"] == 2.5e-3)
    # the rehearsal's shape: a slow revival stays at the cap; its head's error 11.7 times the capped divisor takes
    # most of the lesson without the ceiling, and at most its equal share (1 of 5 senses) with it
    assert slow["capped_until"] == 600 and slow["lesson_share_max"] > 0.5
    assert slow["lesson_share_max_with_ceiling"] <= 0.2 + 1e-9
    assert fast["capped_until"] < 600
    assert T["learning_progress"]["channel_step_at_switch"] > 0


def _stand_ins(S, m, monkeypatch, nf, other=None, ring=None, cluster=(411000, 411050, 411100), watch=0.3):
    """The readers only a pause may run (the checkpoint's model, its companion, the lineage's checkpoints and their
    companions, the stats' collapse watch) replaced by the lite model and stand-in moments: the moments of the OT
    cluster differ in their balance inputs and tokens, as a lived stretch does."""
    moments = {}
    for i, s in enumerate(cluster):
        mo = _moment(m, seed=30 + i)
        moments["cluster_%d" % s] = mo
    monkeypatch.setattr(S, "build", lambda ck, load_optimizer=True, device="cpu": (m, {}, {"step": 411125}))

    def saved(ck, device="cpu", with_ring=True):
        name = str(ck)
        if name in moments:
            mo = moments[name]
            return dict(nf=mo, last_tokens=_target(m, mo), ring=None)
        mo = nf if "earlier" not in name else (other or nf)
        return dict(nf=mo, last_tokens=_target(m, mo), ring=ring if with_ring else None)
    monkeypatch.setattr(S, "saved_moment", saved)
    monkeypatch.setattr(S, "lineage_checkpoints", lambda ck, k, archive=None: [Path("earlier_ckpt_step_411100")][:k])
    monkeypatch.setattr(S, "companion_checkpoints", lambda roots=None, prefix=None: {
        s: Path("cluster_%d" % s) for s in cluster})
    monkeypatch.setattr(S, "collapse_watch_at", lambda steps, experiments=None, prefix=None: [
        {s: watch for s in S.LIVE_SENSES} for _ in steps])
    monkeypatch.setattr(S, "OT_FROM_STEP", 0)
    monkeypatch.setattr(S, "T_PERSISTENCE", (1e-8,))
    monkeypatch.setattr(S, "T_ERROR_MULTIPLE", (11.7,))
    monkeypatch.setattr(S, "T_UPDATES", 300)


def test_one_checkpoint_runs_end_to_end_on_a_stand_in(tmp_path, monkeypatch):
    # everything the pause job runs except the readers only a pause may run, replaced (_stand_ins)
    S = _script()
    m = _with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True))
    nf = _moment(m)
    _stand_ins(S, m, monkeypatch, nf, other=_moment(m, seed=12), ring=torch.randn(8, 50, m.d_model).half())
    rc = S.main(["measure", "--output", str(tmp_path), "--updates", "100", "--moments", "2", "--ot-clusters", "1",
                 "stand_in_checkpoint"])
    doc = json.loads((tmp_path / "vestibular_encoder_repair.json").read_text(encoding="utf-8"))
    row = doc["rows"][0]
    assert rc == 0 and "error" not in row, row.get("error")
    assert row["complete"] is True and row["stage"] == "done"
    assert row["switch"]["switched"] and row["EM_ring_bit_identical"] is True
    assert all(v["within"] for block in row["EM"].values() for v in block.values())
    assert row["EM_dead_token_numbers_varying"]["bfloat16"] == 0
    v = doc["verdicts"]
    assert v["EM"]["met"] and "emulation" in v["LS"]["read_on"] and v["MODE"]["licensed_by_the_card"] is False
    # bf16 RR moves nothing (the lite token is one value in bf16); whether bf16 CORE revives it is what the arm
    # measures, so the candidate follows whichever arm met its line
    assert row["LS"]["RR_bf16"]["readings"]["100"]["spread"] == 0.0
    ls = v["LS"]["per_checkpoint"][0]
    assert ls["LS"] == ls["met_bf16"]["CORE_bf16"] and v["LS"]["met_with_float32_encoder"] is True
    assert v["MODE"]["candidate"] == row["candidate_mode"]
    # RV and OT read BOTH deploy arms (the fourth review's fix round: MODE is decided over every checkpoint read, so
    # the pause's own checkpoint cannot know which arm the deploy runs); judge() takes the chosen mode's arm
    assert row["RV_arms"] == ["CORE_bf16", "CORE32_bf16"]
    assert set(row["RV_rises_by_arm"]) == set(row["OT_verdict_by_arm"]) == {"CORE_bf16", "CORE32_bf16"}
    for arm, rv in row["RV_rises_by_arm"].items():
        assert rv["arm"] == arm and rv["mode"] == S.ARM_MODE[arm] and rv["moments"] == 2
        assert rv["token_precision"] == S.RV_TOKEN_PRECISION[arm]
        assert row["OT_verdict_by_arm"][arm]["deploy"]["precision"] == S.RV_TOKEN_PRECISION[arm]
    ot = v["OT"]
    assert ot.get("predicted_watch_lowest") is not None and ot["clusters"] == 1, ot
    assert ot["arm"] == S.RV_ARMS[v["MODE"]["candidate"]] and ot["precision"] == S.RV_TOKEN_PRECISION[ot["arm"]]
    # the stand-in has no vision (no_vision=True), so its cost holds no vision rise: a cost the kill-line watcher
    # refuses, which names no mode (the fifth review); a model with all four other senses names the candidate
    cost = row["RV_rises_by_arm"][ot["arm"]]["later_reports"]
    complete = all(S._finite_number(cost.get(s)) for s in S.COST_SENSES)
    assert complete is False and "vision" not in cost
    assert v["MODE"]["mode"] == (row["candidate_mode"] if ot["met"] and complete else None)
    if ot["met"]:
        assert "watcher refuses" in v["MODE"]["why"] and "no finite rise for vision" in v["MODE"]["why"]
    if v["MODE"]["mode"] is not None:
        er = v["expected_rise"]
        assert er["moments"] == 2 and set(er["later_reports"]) == set(m.pred_mods) - {"vestibular"}
        assert er["arm"] == S.RV_ARMS[v["MODE"]["mode"]] and er["mode"] == v["MODE"]["mode"]
    assert row["RV_moment_sources"] == ["stand_in_checkpoint", "earlier_ckpt_step_411100"]
    assert doc["rules"].startswith("DECLARED DECISION RULES") and doc["device"] == "cpu"


def test_the_job_writes_its_result_after_every_arm_and_a_stopped_job_keeps_it(tmp_path, monkeypatch):
    # third review, section E: the CPU job wrote its result only after a whole checkpoint, and could not finish one
    # inside its limit; now after every stage and every arm
    S = _script()
    m = _with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True))
    _stand_ins(S, m, monkeypatch, _moment(m))
    real_learn, calls = S.learn, []

    def learn(*a, **kw):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt("the job's time ran out")
        return real_learn(*a, **kw)
    monkeypatch.setattr(S, "learn", learn)
    with pytest.raises(KeyboardInterrupt):
        S.main(["measure", "--output", str(tmp_path), "--updates", "20", "--moments", "1", "--ot-clusters", "0",
                "stand_in_checkpoint"])
    doc = json.loads((tmp_path / "vestibular_encoder_repair.json").read_text(encoding="utf-8"))
    row = doc["rows"][0]
    assert row["complete"] is False and set(row["LS"]) == {"A", "R"} and row["stage"] == "LS"
    assert "EM32" in row and row["T"] and doc["verdicts"]["partial"] == ["stand_in_checkpoint"]


def test_the_cpu_fallback_does_nothing_when_the_cards_result_holds_the_cost(tmp_path, monkeypatch, capsys):
    S = _script()
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"finished": "x", "rows": [{"checkpoint": "stand_in_checkpoint"}],
                                "verdicts": {"expected_rise": {"step": 411125}}}), encoding="utf-8")
    monkeypatch.setattr(S, "build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the fallback ran")))
    assert S.main(["measure", "--output", str(tmp_path / "out"), "--skip-if-result", str(card),
                   "stand_in_checkpoint"]) == 0
    assert "does not run" in capsys.readouterr().out and not (tmp_path / "out" / "vestibular_encoder_repair.json").exists()
    # an unfinished card result, or one about another checkpoint, leaves the fallback to run
    card.write_text(json.dumps({"rows": [{"checkpoint": "stand_in_checkpoint"}],
                                "verdicts": {"expected_rise": {"step": 411125}}}), encoding="utf-8")
    rc = S.main(["measure", "--output", str(tmp_path / "out"), "--skip-if-result", str(card), "stand_in_checkpoint"])
    doc = json.loads((tmp_path / "out" / "vestibular_encoder_repair.json").read_text(encoding="utf-8"))
    assert rc == 1 and "the fallback ran" in doc["rows"][0]["error"]


def test_em32_is_read_whatever_the_mode(tmp_path, monkeypatch):
    # third review, section E: both jobs passed rescale_restart, so the float32 mode's check EM32 never ran and
    # judge() counted it as met; it is read in every run now
    S = _script()
    for mode in ("rescale_restart", "rescale_restart_fp32"):
        m = _with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True))
        nf = _moment(m)
        _stand_ins(S, m, monkeypatch, nf)
        row = S.measure_one("stand_in", 20, mode, moments=0)
        em32 = row["EM32"]
        assert em32["vestibular_token"]["change_max_abs"] > 0            # bf16 before, float32 after
        assert all("share_of_typical_step" in v for k, v in em32.items() if k != "vestibular_token")
        enc = m.encoder_registry.encoders["vestibular"]
        assert "forward" not in enc.__dict__ and not getattr(enc, "vestibular_repair_fp32", False)


def test_the_float32_mode_is_never_chosen_with_em32_unread_or_past_its_line():
    S = _script()
    base = dict(step=411125, checkpoint="c", device="cuda:0", switch=dict(switched=True),
                EM={"float32": {"x": dict(within=True)}}, EM_ring_bit_identical=True,
                LS={"A": {"readings": {"100": {"spread": 1e-6}}}, "RR": {"readings": {"100": {"spread": 0.0}}},
                    "A_bf16": {"readings": {"100": {"spread": 0.0}}},
                    "RR_bf16": {"readings": {"100": {"spread": 0.0}}},
                    "CORE_bf16": {"readings": {"100": {"spread": 0.0}}},
                    "CORE32_bf16": {"readings": {"100": {"spread": 0.2}}}})
    assert S.candidate_mode([base])[0] is None and "not read" in S.candidate_mode([base])[1]
    past = dict(base, EM32={"head_audio": dict(within=False), "vestibular_token": {}})
    assert S.candidate_mode([past])[0] is None and "past its line" in S.candidate_mode([past])[1]
    ok = dict(base, EM32={"head_audio": dict(within=True), "vestibular_token": {}})
    assert S.candidate_mode([ok])[0] == "rescale_restart_fp32"


def test_the_rules_are_applied_as_declared():
    S = _script()

    def arm(spread, hidden=0.0):
        return {"readings": {"100": {"spread": spread, "largest_hidden_step_in_lr": hidden},
                             "200": {"spread": spread, "largest_hidden_step_in_lr": hidden}}}

    def row(step, sw=True, within=True, ring=True, rr16=0.0, core16=0.0, core32=0.2, rr=0.2, hrr=2.0, device="cpu",
            moments=3, em32=True, watch=None, rv_arms=("CORE_bf16", "CORE32_bf16")):
        # what measure_one writes on the pause's own checkpoint since the fourth review's fix round: RV and OT on both
        # deploy arms
        return dict(step=step, checkpoint="c%d" % step, device=device, switch=dict(switched=sw),
                    EM={"float32": {"x": dict(within=within)}, "bfloat16": {"x": dict(within=True)}},
                    EM_ring_bit_identical=ring, EM32={"head_audio": dict(within=em32), "vestibular_token": {}},
                    LS={"A": arm(1e-6), "R": arm(0.2, 7.0), "RR": arm(rr, hrr), "CORE": arm(0.2),
                        "A_bf16": arm(0.0), "RR_bf16": arm(rr16), "CORE_bf16": arm(core16), "CORE32_bf16": arm(core32)},
                    OT_verdict_by_arm={a: {"deploy": {"predicted_watch_lowest": (10.0 * S.LINES["K2_line"]
                                                                                 if watch is None else watch),
                                                      "encoder": a, "precision": S.RV_TOKEN_PRECISION[a]},
                                           "kappa": 0.5, "clusters": 8} for a in S.RV_ARMS.values()},
                    # a rise for each of the four senses the kill-line watcher judges (the fifth review: a cost it
                    # would refuse names no mode)
                    RV_rises_by_arm={a: dict(first_report={"audio": 0.18}, later_reports=dict(RISES),
                                             moments=moments, revived_spread=0.29, arm=a, mode=ARM_MODE.get(a),
                                             token_precision=S.RV_TOKEN_PRECISION.get(a)) for a in rv_arms},
                    RV_moment_sources=["a", "b", "c"])
    # the CPU's emulation: bf16 revives nothing, the float32 encoder does -> rescale_restart_fp32, not licensed by the card
    v = S.judge([row(411125), row(380550), row(344400, sw=False)])
    assert v["population"] == [411125, 380550] and v["EM"]["met"]
    assert v["LS"]["met"] is False and v["LS"]["met_with_float32_encoder"] is True
    assert v["MODE"]["mode"] == "rescale_restart_fp32" and v["MODE"]["licensed_by_the_card"] is False
    assert v["expected_rise"]["later_reports"] == RISES and v["expected_rise"]["moments"] == 3
    assert v["expected_rise"]["arm"] == "CORE32_bf16" and v["expected_rise"]["mode"] == "rescale_restart_fp32"
    # the core's gradient revives it in bf16 (the card): rescale_restart, its cost from CORE_bf16
    v = S.judge([row(411125, core16=0.1, device="cuda:0")])
    assert v["MODE"]["mode"] == "rescale_restart" and v["MODE"]["licensed_by_the_card"] is True
    assert "graphics card" in v["LS"]["read_on"] and v["expected_rise"]["arm"] == "CORE_bf16"
    # a cost read from another arm than the mode's is no cost (third review, section E, semantic), and no cost is no
    # mode (the fourth review's fix round)
    v = S.judge([row(411125, core16=0.1, device="cuda:0", rv_arms=("RR",))])
    assert v["expected_rise"] == {} and v["MODE"]["mode"] is None and "no declared cost" in v["MODE"]["why"]
    # revives in bf16 only with hidden steps past a fresh state's range: the float32 encoder mode
    assert S.judge([row(411125, rr16=0.1, hrr=4.5)])["MODE"]["mode"] == "rescale_restart_fp32"
    # the float32 mode with its precision check past its line: no deploy
    assert S.judge([row(411125, em32=False)])["MODE"]["mode"] is None
    # OT (the token over time, third review, section C) under K2's line: the dead-token stop would stop the
    # revival, so no deploy; unread: no deploy either
    v = S.judge([row(411125, core16=0.1, device="cuda:0", watch=0.5 * S.LINES["K2_line"])])
    assert v["MODE"]["mode"] is None and v["MODE"]["candidate"] == "rescale_restart" and not v["OT"]["met"]
    assert "dead-token stop" in v["MODE"]["why"] and v["expected_rise"] == {}
    r = row(411125, core16=0.1, device="cuda:0")
    r.pop("OT_verdict_by_arm")
    assert S.judge([r])["MODE"]["mode"] is None
    # nothing revives: no deploy, no cost
    v = S.judge([row(411125, core32=0.0, rr=0.0)])
    assert v["MODE"]["mode"] is None and v["expected_rise"] == {}
    assert not S.judge([row(411125, within=False)])["EM"]["met"]
    assert not S.judge([row(411125, ring=False)])["EM"]["met"]
    assert S.judge([dict(checkpoint="x", error="boom")])["failed"][0]["error"] == "boom"


def _mode_plant_rows(S, core_bf16_spanning, both_arms, core32_watch=None):
    """The fourth review's planted case (vrev4/mode_plant.py): the pause's checkpoint (415,075) revives in bf16 (its
    own readings choose rescale_restart); a spanning checkpoint (347,791) revives in bf16 or only with the float32
    encoder. both_arms=False: the pause's row as the third fix round's measure_one wrote it (RV and OT on the arm its
    own readings chose); True: as it writes it now (both deploy arms)."""
    def arms(core_bf16, core32_bf16):
        def r(spread, hidden=2.0):
            return {"readings": {"100": {"spread": spread, "largest_hidden_step_in_lr": hidden}}}
        return {"A": r(2e-5), "R": r(0.3), "RR": r(0.19), "R8": r(0.1), "CORE": r(0.083),
                "A_bf16": r(0.0), "RR_bf16": r(0.0), "CORE_bf16": r(core_bf16), "CORE32_bf16": r(core32_bf16)}

    def row(step, core_bf16, core32_bf16):
        ok = {"within": True}
        return dict(checkpoint="ck_%d" % step, step=step, device="cuda:0", complete=True, switch={"switched": True},
                    EM={"float32": {"head_audio": ok}, "bfloat16": {"head_audio": ok}}, EM_ring_bit_identical=True,
                    EM32={"head_audio": {"within": True, "share_of_typical_step": 0.004}, "vestibular_token": {}},
                    LS=arms(core_bf16, core32_bf16))
    own = row(415075, 0.08, 0.12)
    rises = lambda arm: dict(first_report={"audio": 0.3}, later_reports={"audio": 0.004, "vision": 0.001,   # noqa
                                                                         "tactile": 0.001, "proprioception": 0.001},
                             moments=3, arm=arm, mode=ARM_MODE[arm], token_precision=S.RV_TOKEN_PRECISION[arm],
                             revived_spread=0.08)
    ot = lambda arm, low=0.03: dict(deploy=dict(encoder=arm, precision=S.RV_TOKEN_PRECISION[arm],  # noqa: E731
                                                predicted_watch_lowest=low), kappa=1.09, clusters=6)
    cand, _ = S.candidate_mode([dict(own)])
    assert cand == "rescale_restart"
    if both_arms:
        own.update(candidate_mode=cand, RV_arms=list(S.RV_ARMS.values()),
                   RV_rises_by_arm={a: rises(a) for a in S.RV_ARMS.values()},
                   OT_verdict_by_arm={a: ot(a, core32_watch if (a == "CORE32_bf16" and core32_watch) else 0.03)
                                      for a in S.RV_ARMS.values()})
    else:
        arm = S.RV_ARMS[cand]
        own.update(candidate_mode=cand, RV_arm=arm, RV_rises=rises(arm),
                   OT_verdict=dict(ot(arm), deploy=dict(ot(arm)["deploy"], encoder="deploy")))
    return [own, row(347791, core_bf16_spanning, 0.11)]


def test_the_reviews_mode_plant_is_refused():
    # the fourth review (section 1, MAJOR; vrev4/mode_plant.py through the branch's own judge): the pause's checkpoint
    # revives in bf16, a spanning checkpoint only with the float32 encoder; MODE became rescale_restart_fp32 with OT
    # "met" on the bf16 arm and no declared cost, and the deploy would have run with its lines on the other senses never
    # judged. Now a mode whose arm holds no OT or no cost is no mode
    S = _script()
    v = S.judge(_mode_plant_rows(S, 0.03, both_arms=False))
    assert v["MODE"]["candidate"] == "rescale_restart_fp32" and v["MODE"]["mode"] is None, v["MODE"]
    assert v["OT"]["met"] is False and v["OT"]["arm"] == "CORE32_bf16" and "CORE_bf16 only" in v["OT"]["why"]
    assert "no declared cost" in v["MODE"]["why"] and "CORE32_bf16" in v["MODE"]["why"] and v["expected_rise"] == {}
    # with every checkpoint reviving in bf16 the same rows give rescale_restart and its cost, as before
    v = S.judge(_mode_plant_rows(S, 0.07, both_arms=False))
    assert v["MODE"]["mode"] == "rescale_restart" and v["expected_rise"]["arm"] == "CORE_bf16"


def test_the_float32_mode_takes_ot_and_its_cost_from_its_own_arm():
    # RV and OT read on both deploy arms of the pause's checkpoint: the float32 mode chosen over every checkpoint gets
    # OT read on CORE32_bf16 in float32 (the precision that arm's token is computed in) and the cost from CORE32_bf16
    S = _script()
    v = S.judge(_mode_plant_rows(S, 0.03, both_arms=True))
    assert v["MODE"]["mode"] == "rescale_restart_fp32" and v["MODE"]["arm"] == "CORE32_bf16", v["MODE"]
    assert v["OT"]["met"] and v["OT"]["arm"] == "CORE32_bf16" and v["OT"]["precision"] == "float32"
    er = v["expected_rise"]
    assert er["arm"] == "CORE32_bf16" and er["mode"] == "rescale_restart_fp32" and er["token_precision"] == "float32"
    # the float32 arm's own OT under K2's line: no deploy, whatever the bf16 arm read
    v = S.judge(_mode_plant_rows(S, 0.03, both_arms=True, core32_watch=0.5 * S.LINES["K2_line"]))
    assert v["MODE"]["mode"] is None and "dead-token stop" in v["MODE"]["why"] and v["expected_rise"] == {}
    # an OT verdict read in another precision than the arm's token is not read
    rows = _mode_plant_rows(S, 0.03, both_arms=True)
    rows[0]["OT_verdict_by_arm"]["CORE32_bf16"]["deploy"]["precision"] = "bfloat16"
    v = S.judge(rows)
    assert v["MODE"]["mode"] is None and "read in bfloat16" in v["OT"]["why"]


def test_the_cpu_fallback_does_nothing_when_the_card_read_the_checkpoint_to_its_end(tmp_path, monkeypatch, capsys):
    # the fourth review (minor): a card verdict of no mode (a do-not-deploy reading), or a card job stopped by its limit
    # after the pause's own checkpoint, triggered the whole CPU run, which could then print a MODE and a cost of its own
    S = _script()
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"rows": [{"checkpoint": "stand_in_checkpoint", "complete": True},
                                         {"checkpoint": "spanning", "complete": False}],
                                "verdicts": {"MODE": {"mode": None}, "expected_rise": {}}}), encoding="utf-8")
    monkeypatch.setattr(S, "build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the fallback ran")))
    assert S.main(["measure", "--output", str(tmp_path / "out"), "--skip-if-result", str(card),
                   "stand_in_checkpoint"]) == 0
    assert "read the checkpoint to its end" in capsys.readouterr().out
    assert not (tmp_path / "out" / "vestibular_encoder_repair.json").exists()


# ------------------------------------------------------------------------------ OT: the token over time (review 3, C)
def test_the_clusters_are_saved_moments_within_the_collapse_watchs_span():
    S = _script()
    found = {s: Path("c%d" % s) for s in (380000, 390000, 390025, 390600, 391100, 395000, 395050, 395100, 395900,
                                          396500, 411125)}
    got = [[s for s, _p in c] for c in S.ot_clusters(411125, found, from_step=382800)]
    # 390,000-390,600 (3 within 1,000 steps; 391,100 is 1,100 after the first), 395,000-395,900 (4); 380,000 is
    # before the stretch, 396,500 and 411,125 have no neighbours
    assert got == [[390000, 390025, 390600], [395000, 395050, 395100, 395900]]
    assert [[s for s, _p in c] for c in S.ot_clusters(411125, found, from_step=382800, limit=1)] == [
        [395000, 395050, 395100, 395900]]


def test_the_collapse_watch_statistic_is_the_models_own():
    S = _script()
    m = _dead_lite()
    g = torch.Generator().manual_seed(2)
    ring = [torch.randn(m.d_model, generator=g) for _ in range(9)]
    for t in ring:
        m.token_ring["vestibular"].append(t)
    assert S.watch_statistic(ring) == pytest.approx(m.token_temporal_std()["vestibular"], rel=1e-6)


def test_the_token_over_time_reads_the_revived_and_the_dead_encoder_on_each_cluster(monkeypatch):
    S = _script()
    m = _with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True))
    nf = _moment(m)
    _stand_ins(S, m, monkeypatch, nf, cluster=(411000, 411050, 411100, 411200))
    enc = m.encoder_registry.encoders["vestibular"]
    dead = {n: p.detach().clone() for n, p in enc.named_parameters()}
    VS.repair_encoder(m, load_optimizer=True, restart_moments=True)
    o = S.learn(m, nf["_nf_raw"]["vestibular"], 60, "float32", keep_at=(60,))
    clusters = S.ot_clusters(411300, S.companion_checkpoints(), from_step=0)
    ot = S.over_time(m, {"dead": dead, "RR@60": o["snapshots"][60]}, clusters)
    row = ot["clusters"][0]
    assert row["steps"] == [411000, 411050, 411100, 411200] and row["bodies"] == 8
    # the dead token does not change over time in either precision; the revived one does, in both
    assert max(row["vestibular"]["dead"]["bfloat16"]) == 0.0 and max(row["vestibular"]["dead"]["float32"]) < 1e-5
    assert min(row["vestibular"]["RR@60"]["float32"]) > 1e-3
    live = set(S.LIVE_SENSES) & set(m.active)
    assert set(row["live"]) == live and min(row["live"]["audio"]) > 0.1
    # the encoder is put back as it was before the reading
    assert all(torch.equal(p, o["snapshots"][60][n]) for n, p in enc.named_parameters())
    watch = [{s: 0.3 for s in S.LIVE_SENSES}]
    v = S.ot_verdict(ot, watch, deploy_label="RR@60", deploy_precision="float32")
    kappa = v["kappa"]
    assert kappa == pytest.approx(__import__("statistics").median([0.3 / row["live"][s][0] for s in live]))
    low = v["by_encoder"]["RR@60"]["float32"]["predicted_watch_lowest"]
    assert low == pytest.approx(row["vestibular"]["RR@60"]["float32"][0] * kappa)
    assert v["by_encoder"]["RR@60"]["K2_line"] == pytest.approx((S.DEAD_WATCH_LARGEST * low) ** 0.5)
    assert v["by_encoder"]["RR@60"]["A2_line"] == low and v["deploy"]["predicted_watch_lowest"] == low
    assert v["by_encoder"]["dead"]["bfloat16"]["predicted_watch_lowest"] == 0.0


def test_the_declared_k2_line_is_the_watchers():
    S = _script()
    spec = importlib.util.spec_from_file_location("vestibular_repair_watch", SCRIPT.parent / "vestibular_repair_watch.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    assert S.LINES["K2_line"] == W.LINES["K2_line"] and S.LINES["K2_line"] > S.DEAD_WATCH_LARGEST


# ------------------------------------------------------------------------------ the lesson weight (review 3, D)
def test_the_core_arm_and_the_transient_read_the_lesson_weight_actually_used():
    # third review, section D: T and the CORE arm left the sense's lesson weight out; at a weight of 2 the lesson the
    # optimizer sees gave the sense 33 in 100 while they (and the gauge) said 20
    S = _script()
    for w in (1.0, 2.0):
        m = _with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True,
                                      vestibular_lesson_weight=w))
        assert S.lesson_weight_of(m) == w
        T = S.transients(m, P_values=(1e-8,), multiples=(11.7,), updates=300)
        assert T["lesson_weight"] == w and T["rows"][0]["lesson_share_max_with_ceiling"] <= 0.2 + 1e-9
        nf = _moment(m)
        cw = S.core_weights(m, nf, _target(m, nf), "float32")
        assert cw["lesson_weight"] == w
    # without the ceiling, the weight shows in T's share: the sense takes more of the lesson at 2 than at 1
    one = S.transients(_with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True,
                                                 vestibular_lesson_weight=1.0)),
                       P_values=(1e-8,), multiples=(11.7,), updates=50)
    two = S.transients(_with_averages(_dead_lite(next_frame_norm="persistence", learning_progress=True,
                                                 vestibular_lesson_weight=2.0)),
                       P_values=(1e-8,), multiples=(11.7,), updates=50)
    assert two["rows"][0]["lesson_share_max"] > one["rows"][0]["lesson_share_max"]


def test_the_rehearsals_share_line_fails_a_report_that_carries_no_share():
    # third review, minor: R1 passed a report whose share was missing
    spec = importlib.util.spec_from_file_location("vestibular_repair_rehearsal",
                                                  SCRIPT.parent / "vestibular_repair_rehearsal.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)

    def rep(share, replay_share=0.1):
        return dict(step=1, share=share, replay_share=replay_share, graded=["audio", "vestibular"],
                    replay_graded=["audio", "vestibular"], between=0.1, ratios={})
    assert R.judge([rep(0.4)], [rep(0.4)])["R1"]["met"] is True
    assert R.judge([rep(None)], [rep(0.4)])["R1"]["met"] is False
    assert R.judge([rep(0.4, None)], [rep(0.4)])["R1"]["met"] is False
    assert R.judge([rep(0.6)], [rep(0.4)])["R1"]["met"] is False


def test_a_saved_moment_whose_companion_is_gone_is_left_out_of_the_clusters(tmp_path):
    S = _script()
    root = tmp_path / "experiments"
    for step, present in ((411000, True), (411050, False), (411100, True)):
        ck = root / "H84_fresh_drivers_20260918_segX" / "checkpoints" / ("ckpt_step_%d" % step)
        ck.mkdir(parents=True)
        comp = tmp_path / ("companion_%d.pt.gz" % step)
        if present:
            comp.write_bytes(b"x")
        (ck / "episode_state.json").write_text(json.dumps({"storage_path": str(comp), "sha256": "0" * 64}),
                                               encoding="utf-8")
    assert sorted(S.companion_checkpoints(roots=(root,))) == [411000, 411100]
    assert sorted(S.companion_checkpoints(roots=(root,), available=False)) == [411000, 411050, 411100]


def test_k2_and_a2_are_drawn_from_the_committed_over_time_measurement():
    # third review, finding C: K2 and A2 read body 0's token over time, and nothing before the deploy measured it; the
    # lines are the declared rule's output on OT's measurement on 415,075 (the CORE arm, 100 updates, bf16 tokens)
    S = _script()
    doc = json.loads((HERE.parents[4] / "docs" / "gain_of_function_v3" / "measurements" /
                      "vestibular_encoder_over_time_415075_2026-09-26.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("vestibular_repair_watch", SCRIPT.parent / "vestibular_repair_watch.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    again = S.ot_verdict(doc["OT"], doc["collapse_watch_live"])        # the rule reproduces from the stored readings
    core = again["by_encoder"]["CORE@100"]
    assert core["bfloat16"]["predicted_watch_lowest"] > 0.0            # the bf16 reading itself, not the fallback
    assert W.LINES["K2_line"] == pytest.approx(core["K2_line"], rel=5e-3)
    assert W.LINES["A2_line"] == pytest.approx(core["A2_line"], rel=5e-3)
    assert W.LINES["dead_watch_largest"] == S.DEAD_WATCH_LARGEST
    assert again["by_encoder"]["dead"]["bfloat16"]["predicted_watch_lowest"] == 0.0 and again["clusters"] >= 3


def test_a_cost_the_watcher_refuses_names_no_mode():
    # the fifth review (section 2, MAJOR): judge() named a mode when the arm held any cost, while the kill-line
    # watcher's load_cost refuses a cost read on fewer than 2 saved moments or missing a finite rise for any of the
    # four other senses; the deploy would then run its whole watch with no stop line judging vision, touch, hearing or
    # body position. Planted: 1 moment, no hearing rise, a touch rise that is not a number
    S = _script()
    spec = importlib.util.spec_from_file_location("vestibular_repair_watch_for_cost", SCRIPT.parent /
                                                  "vestibular_repair_watch.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    assert tuple(S.COST_SENSES) == tuple(W.OTHER_SENSES) and S.LINES["min_moments"] == W.LINES["min_moments"]

    def planted(change):
        rows = _mode_plant_rows(S, 0.03, both_arms=True)
        change(rows[0]["RV_rises_by_arm"]["CORE32_bf16"])
        return S.judge(rows)

    v = planted(lambda rv: None)
    assert v["MODE"]["mode"] == "rescale_restart_fp32" and v["expected_rise"]["moments"] == 3
    v = planted(lambda rv: rv.update(moments=1))
    assert v["MODE"]["mode"] is None and v["expected_rise"] == {}
    assert "watcher refuses" in v["MODE"]["why"] and "1 saved moment(s), fewer than 2" in v["MODE"]["why"]
    v = planted(lambda rv: rv["later_reports"].pop("audio"))
    assert v["MODE"]["mode"] is None and "no finite rise for audio" in v["MODE"]["why"]
    for bad in (float("nan"), "0.001", None, True):
        v = planted(lambda rv: rv["later_reports"].update(tactile=bad))
        assert v["MODE"]["mode"] is None and "no finite rise for tactile" in v["MODE"]["why"], bad
    v = planted(lambda rv: rv.update(moments=True))
    assert v["MODE"]["mode"] is None
