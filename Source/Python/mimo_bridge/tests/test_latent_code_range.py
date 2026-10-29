# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Trial codes inside the encoder's own code range (lem.latent_body.code_range, Oct 1, 2026): the readings tools, the
causal-control test, the planner search and the whole-body pause-job driver. CPU, no world (stand-in sessions).

The defect these tests guard: trial codes drawn per dimension within 2 standard deviations (a box) lay a median
4.9-26.6 Mahalanobis units from the whole-body fit's own codes (2.2-2.5 for those), where the decoder has learned
nothing (agents/latent_neighbor_commands_2026-10-01.md)."""
import json
import types

import numpy as np
import pytest
import torch

from lem.latent_body.code_range import CodeRange, resolve_sampling
from lem.latent_body.learning import ShadowLearner
from lem.latent_body.tree import full_body_tree
from mimo_bridge.tests.test_latent_full_body_rows import CONTRACT, NAMES, _example, config


def _cloud(n=3000, seed=0):
    """Codes with strongly correlated dimensions (correlation of the first two 0.95), as the fit's are."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.tensor([[1.0, 0.95, 0.0, 0.2], [0.0, 0.31, 0.0, 0.1], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.3]])
    return torch.randn(n, 4, generator=g) @ mix + torch.tensor([0.5, -0.2, 0.1, 0.0])


def _reference(z):
    x = z.double().numpy()
    m, inv = x.mean(0), np.linalg.inv(np.cov(x.T))
    return np.sqrt(np.einsum("ni,ij,nj->n", x - m, inv, x - m))


# ------------------------------------------------------------------ the measured range
def test_distance_and_range_are_the_encoders_own_mahalanobis_quantile():
    # Defects caught: a wrong covariance (its diagonal only: correlated dimensions measured as independent); a range
    # other than the 95th percentile of the encoder's own distances.
    z = _cloud()
    cr = CodeRange({"p": z})
    d = _reference(z)
    assert np.allclose(cr.distance("p", z).numpy(), d, atol=1e-4)
    assert cr.range["p"] == pytest.approx(float(np.quantile(d, .95)), abs=1e-4)
    assert float(np.median(cr.own["p"].numpy())) == pytest.approx(float(np.median(d)), abs=1e-4)
    # a code along the cloud's long axis is near; the same size across it is far
    along, across = torch.tensor([[1.0, 0.95, 0, 0]]) + z.mean(0), torch.tensor([[1.0, -0.95, 0, 0]]) + z.mean(0)
    assert float(cr.distance("p", across)) > 5 * float(cr.distance("p", along))
    with pytest.raises(ValueError, match="at least"):
        CodeRange({"p": z[:20]})


def test_encoded_codes_are_real_codes_inside_the_range_and_never_the_rows_own():
    # Defects caught: sampling outside the range (every real code offered, the farthest 5% included); a row's own code
    # offered as another moment's.
    z = _cloud(400)
    cr = CodeRange({"p": z})
    g = torch.Generator().manual_seed(1)
    codes = cr.sample("p", 4000, "encoded", g)
    match = (codes[:, None, :] == z.float()[None]).all(-1)
    assert match.any(1).all()                                   # every trial code is a code the encoder gave
    assert (cr.distance("p", codes) <= cr.range["p"] + 1e-9).all()
    assert cr.report("p", codes)["beyond_range"] == 0.0
    own = int(cr.pool["p"][0])
    codes = cr.sample("p", 4000, "encoded", g, exclude=own)
    assert not (codes == z[own].float()).all(-1).any()
    with pytest.raises(ValueError, match="box"):
        cr.sample("p", 3, "box", g)


def test_gaussian_codes_keep_the_fitted_covariance_inside_the_range():
    # Defects caught: Gaussian draws not rejected beyond the range; drawn without the fitted correlation.
    z = _cloud()
    cr = CodeRange({"p": z})
    codes = cr.sample("p", 6000, "gaussian", torch.Generator().manual_seed(2))
    assert codes.shape == (6000, 4)
    assert (cr.distance("p", codes) <= cr.range["p"] + 1e-9).all()
    fitted = np.corrcoef(z.numpy().T)[0, 1]
    drawn = np.corrcoef(codes.numpy().T)[0, 1]
    assert fitted > 0.9 and abs(drawn - fitted) < 0.03


def test_box_codes_lie_beyond_the_range_of_correlated_codes():
    # The fact the modes exist for: per-dimension draws within 2 standard deviations reach corners the encoder never
    # visits; the report measures it.
    z = _cloud()
    cr = CodeRange({"p": z})
    rng = np.random.default_rng(0)
    box = z.mean(0) + torch.from_numpy(rng.uniform(-2, 2, size=(2000, 4))).float() * z.std(0)
    assert cr.report("p", box)["beyond_range"] > 0.5
    assert cr.report("p", box)["trial_median"] > 2 * cr.report("p", z)["trial_median"]


def test_auto_is_encoded_for_the_whole_body_and_box_for_the_pilot():
    # Defect caught: the old box left as the whole-body default.
    from mimo_bridge.tests.latent_body_fixtures import config as pilot_config
    assert resolve_sampling("auto", config()) == "encoded"
    assert resolve_sampling("auto", pilot_config()) == "box"
    assert resolve_sampling("gaussian", pilot_config()) == "gaussian"
    with pytest.raises(ValueError):
        resolve_sampling("corner", config())


# ------------------------------------------------------------------ the readings tools
def _learner():
    return ShadowLearner(full_body_tree(NAMES), config())


def _examples(n=60):
    return [_example(k) for k in range(n)]


def test_part_readings_draw_other_rows_codes_and_record_the_mode(monkeypatch):
    # Defects caught: box codes drawn under the encoded label; the mode not recorded; a code outside the range.
    import mimo_bridge.tools.latent_body_intervention as LI
    from mimo_bridge.tools.latent_body_part_readings import part_readings
    learner, examples = _learner(), _examples()
    with torch.no_grad():
        z = learner.model.encode(learner.batch(examples))
    cr = CodeRange(z)
    seen = []
    real = LI.decode_with

    def spy(model, state, recent, roots, override):
        seen.append((dict(roots), dict(override)))
        return real(model, state, recent, roots, override)
    monkeypatch.setattr(LI, "decode_with", spy)
    parts = ["trunk", "arm_L", "index_L"]
    r = part_readings(learner, examples, every=12, codes=4, parts=parts, code_sampling="auto")
    assert r["code_sampling"] == "encoded" and r["code_sampling_requested"] == "auto"
    for name in parts:
        p = r["parts"][name]
        assert p["code_sampling"] == "encoded" and p["code_range"]["beyond_range"] == 0.0
        assert p["code_range"]["trial_max"] <= p["code_range"]["range"] + 1e-6
        assert p["code_range"]["trial_codes"] == 3 * p["rows_sampled"]
    pools = {n: z[n][cr.pool[n]] for n in parts}
    for roots, override in seen:
        name, code = next(iter(override.items())) if override else ("trunk", roots["trunk"])
        if name not in pools:
            continue
        trial = code[1:]
        assert torch.allclose(code[0], z[name].mean(0), atol=1e-6)
        assert (trial[:, None, :] == pools[name][None]).all(-1).any(1).all()
    box = part_readings(learner, examples, every=12, codes=4, parts=parts, code_sampling="box")
    assert box["code_sampling"] == "box" and all(p["code_sampling"] == "box" for p in box["parts"].values())


def _unbatched_direction(learner, examples, part, every, codes, seed):
    """The readings tool before Oct 1 evening: one code at a time, the same box draws."""
    from lem.latent_body.trajectory import row_options, row_scales
    from mimo_bridge.tools.latent_body_intervention import decode_with
    from mimo_bridge.tools.latent_body_part_readings import _corr, graded_columns
    tree, cfg, model = learner.tree, learner.config, learner.model
    names, rows = tree.handle_names, row_options(cfg)
    batch = learner.batch(examples)
    with torch.no_grad():
        z = model.encode(batch)
        roots = {r: z[r].mean(0, keepdim=True) for r in tree.roots}
        rng = np.random.default_rng(seed)
        node, own, sub = tree.by_name[part], list(tree.owned[part]), list(tree.subtree[part])
        _, cols = graded_columns(tree, part, rows["body"])
        sa = row_scales(names, own, cfg["scales"], "action", rows["digit_scales"], rows["body"])
        sc = row_scales(names, sub, cfg["scales"], "change", rows["digit_scales"], rows["body"])
        centre, spread = z[part].mean(0, keepdim=True), z[part].std(0, keepdim=True)
        agree = []
        for r in range(0, len(examples), every):
            state = {k: v[r][None] for k, v in batch.state.items()}
            recent = {k: v[r][None] for k, v in batch.history.items()}
            pc, tg = [], []
            for k in range(codes):
                u = (torch.zeros(1, model.latent_dim) if k == 0 else
                     torch.from_numpy(rng.uniform(-2, 2, size=(1, model.latent_dim)).astype(np.float32)))
                code = centre + u * spread
                preds = (decode_with(model, state, recent, dict(roots, **{part: code}), {}) if node.parent is None
                         else decode_with(model, state, recent, roots, {part: code}))
                change = preds[part].change.reshape(len(sub), 6) * sc
                pc.append(change[[sub.index(i) for i in own]][:, cols].flatten())
                tg.append((preds[part].action.reshape(len(own), 8) * sa)[:, cols].flatten())
            agree.append(_corr(torch.stack(pc), torch.stack(tg)))
    return float(np.median(agree))


def test_box_readings_repeat_the_earlier_one_code_at_a_time_tool():
    # Defect caught: the batched tool drawing the box codes in another order (the earlier numbers would not repeat).
    from mimo_bridge.tools.latent_body_part_readings import part_readings
    learner, examples = _learner(), _examples()
    r = part_readings(learner, examples, every=12, codes=5, seed=3, parts=["trunk", "index_R"], code_sampling="box")
    for part in ("trunk", "index_R"):
        assert r["parts"][part]["direction_agreement"] == pytest.approx(
            _unbatched_direction(learner, examples, part, 12, 5, 3), abs=1e-3)


def test_index_readings_record_the_mode_and_the_range():
    from mimo_bridge.tools.latent_body_index_readings import index_readings
    learner, examples = _learner(), _examples()
    r = index_readings(learner, examples, every=12, codes=4)
    assert r["code_sampling"] == "encoded"
    assert set(r["code_range"]) == {"index_L", "index_R"}
    assert all(v["beyond_range"] == 0.0 for v in r["code_range"].values())
    assert index_readings(learner, examples, every=12, codes=4, code_sampling="box")["code_sampling"] == "box"


# ------------------------------------------------------------------ the causal-control test
def test_the_causal_test_executes_only_encoded_codes_and_records_them(monkeypatch):
    # Defects caught: box codes executed under the encoded label; the mode not recorded in a part's reading; encoded
    # sampling accepted without a measured range.
    import mimo_bridge.latent_body_runtime as RT
    import mimo_bridge.tools.latent_body_intervention as LI
    from mimo_bridge.tests.test_latent_full_body_tools import _Session
    monkeypatch.setattr(RT, "observed_contract", lambda frame: CONTRACT)
    learner, examples = _learner(), _examples()
    with torch.no_grad():
        z = learner.model.encode(learner.batch(examples))
    cr = CodeRange(z)
    centres = {n: z[n].mean(0, keepdim=True) for n in z}
    spread = {n: z[n].std(0) for n in z}
    seen = []
    real = LI.decode_with

    def spy(model, state, recent, roots, override):
        seen.append((dict(roots), dict(override)))
        return real(model, state, recent, roots, override)
    monkeypatch.setattr(LI, "decode_with", spy)
    args = types.SimpleNamespace(warmup_steps=2, codes=4, centre="mean", seed=0, code_sampling="encoded")
    out = LI.test_body(_Session(), learner, spread, args, np.random.default_rng(0), centres, cr)
    assert len(out) == 24
    for name, p in out.items():
        assert p["code_sampling"] == "encoded" and p["code_range"]["beyond_range"] == 0.0
    pools = {n: z[n][cr.pool[n]] for n in z}
    roots = set(learner.tree.roots)
    for r, override in seen:
        for name, code in (override.items() if override else [(n, c) for n, c in r.items() if n in roots]):
            if torch.allclose(code, centres[name]):
                continue
            assert (code == pools[name]).all(-1).any(), name
    with pytest.raises(ValueError, match="range"):
        LI.test_body(_Session(), learner, spread, args, np.random.default_rng(0), centres)
    box = LI.test_body(_Session(), learner, spread, types.SimpleNamespace(warmup_steps=2, codes=4, centre="mean",
                                                                        seed=0), np.random.default_rng(0), centres, cr)
    assert all(p["code_sampling"] == "box" for p in box.values())


# ------------------------------------------------------------------ the planner search
def test_the_search_draws_whole_real_rows_of_root_codes(monkeypatch):
    # Defects caught: each root drawn from a different row (a combination the encoder never gave); the zero code kept
    # as candidate 0; encoded sampling accepted without a measured range; the box search changed.
    from lem.planner.latent_candidates import search
    from mimo_bridge.tests.latent_body_fixtures import config as pilot_config, example, tree
    learner = ShadowLearner(tree(), pilot_config())
    examples = [example(s) for s in range(1, 61)]
    batch = learner.batch(examples)
    with torch.no_grad():
        z = learner.model.encode(batch)
    roots = learner.tree.roots
    cr = CodeRange({r: z[r] for r in roots})
    one = learner.batch([examples[0]])
    seen = {}
    real = learner.model.decode

    def spy(state, history, intentions):
        seen.setdefault("roots", intentions)
        return real(state, history, intentions)
    monkeypatch.setattr(learner.model, "decode", spy)
    score = lambda p: p["index_L"].change.square().mean(-1)
    choice = search(learner.model, one.state, one.history, score, candidates=12, generator=torch.Generator().manual_seed(0),
                    sampling="encoded", code_range=cr)
    drawn = seen["roots"]
    for r in roots:
        assert torch.allclose(drawn[r][0], z[r].mean(0))
    joint = torch.cat([z[r] for r in roots], -1)
    for k in range(1, 12):
        row = torch.cat([drawn[r][k] for r in roots])
        assert (row == joint).all(-1).any()
    assert choice.sampling == "encoded" and set(choice.code_range) == set(roots)
    assert all(v["beyond_range"] == 0.0 and "selected" in v for v in choice.code_range.values())
    with pytest.raises(ValueError, match="range"):
        search(learner.model, one.state, one.history, score, candidates=4, generator=torch.Generator(),
               sampling="encoded")
    seen.clear()
    box = search(learner.model, one.state, one.history, score, candidates=6, radius=.5,
                 generator=torch.Generator().manual_seed(0))
    assert box.sampling == "box" and all(seen["roots"][r][0].abs().sum() == 0 for r in roots)
    assert all(seen["roots"][r].abs().max() <= .5 for r in roots)


def test_the_comparison_arm_searches_with_its_declared_mode():
    # Defects caught: a latent arm searching the box although its settings say encoded; the range measured on too few
    # rows.
    from mimo_bridge.tests.test_latent_full_body_rows import _obs
    from mimo_bridge.tools import latent_body_compare as LC
    learner, examples = _learner(), _examples()
    spec = LC.latent_arm(learner, examples)
    assert spec["sampling"] == "encoded" and set(spec["code_range"].mean) == set(learner.tree.roots)
    ex = _example(2)
    ex.history, ex.start = _obs(0, 4), _obs(1, 5)
    template = torch.zeros(26, 9)
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    _, _, receipt = LC.latent_order(learner, ex.history, ex.start, ex.start.bones[:, 4:7].numpy() + 1, template,
                                    candidates=6, generator=torch.Generator().manual_seed(0), **LC.search_settings(spec))
    assert receipt["code_sampling"] == "encoded"
    assert set(receipt["candidate_code_range"]) == set(learner.tree.roots)
    assert all(v["beyond_range"] == 0.0 for v in receipt["candidate_code_range"].values())
    with pytest.raises(ValueError, match="50"):
        LC.measure_code_range(learner, examples[:20])


# ------------------------------------------------------------------ the pause-job driver
def _driver():
    from mimo_bridge.tests.test_latent_compare_design import _driver as load
    return load()


def test_the_pause_plan_fits_with_the_digit_switch_and_tests_on_encoded_codes(tmp_path):
    # Defects caught: a fit without consequence_from_command; a causal test, hold causal test, reading or comparison
    # launched without the encoded codes; a result accepted without the mode recorded; the shadow switch reading the
    # fit without the key.
    from pathlib import Path
    D = _driver()
    assert D.CONSEQUENCE == ["digit"] and D.CODE_SAMPLING == "encoded"
    split = tmp_path / "split"
    split.mkdir()
    base = dict(config(), seed=7)
    (split / "config.json").write_text(json.dumps(base), encoding="utf-8")
    written = D.write_switch_config(split, tmp_path / "run" / "config.json")
    got = json.loads(written.read_text(encoding="utf-8"))
    assert got == dict(base, consequence_from_command=["digit"])
    ShadowLearner(full_body_tree(NAMES), got)                              # the key validates on the whole body
    with pytest.raises(SystemExit):
        D.switch_config(dict(base, consequence_from_command=["leg"]))
    sampled = lambda cmd: cmd[cmd.index("--code-sampling") + 1] == "encoded"
    assert sampled(D.causal_cmd("ck", "h.pt", "t.jsonl", "out"))
    assert sampled(D.readings_cmd("h.pt", "hold.jsonl", "out.json"))
    assert D.readings_cmd("h.pt", "hold.jsonl", "out.json")[-4:-2] == ["--every", 3]
    assert sampled(D.compare_cmd("ck", dict(primary=("h.pt", "hold.jsonl"))))
    assert D.sampling_of(dict(code_sampling="encoded"), "x") == "encoded"
    for bad in ({}, dict(code_sampling="box")):
        with pytest.raises(SystemExit):
            D.sampling_of(bad, "x")
    assert D.RUN.parent == D.OUT and D.TEST.name.endswith("digit_cmd")
    # the comparison takes its fits from RUN and its bank and held-out rows from OUT's splits
    out, data = tmp_path / "out", tmp_path / "data"
    (out / "fit").mkdir(parents=True)
    (out / "fit" / "hierarchy.pt").write_bytes(b"x")
    (out / "verdict.json").write_text(json.dumps(dict(lines=dict(no_regression=dict(passed=True)))))
    (data / "hold_split").mkdir(parents=True)
    (data / "hold_split" / "witness_bank.pt").write_bytes(b"x")
    plan, _ = D.compare_plan(out, data)
    assert plan["primary"] == (out / "fit" / "hierarchy.pt", data / "split" / "holdout.jsonl")
    # the shadow switch moves the fit the verdict judged (RUN), never the fit without the key
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "switch", Path(D.__file__).with_name("latent_shadow_full_body_switch_2026-10-01.py"))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    assert S.VERDICT == D.RUN / "verdict.json"
    jobs = json.loads((Path(D.__file__).resolve().parents[1] / "pause_jobs_2026-10-01_latent_full_body.json").read_text(
        encoding="utf-8"))
    assert [job["cmd"][job["cmd"].index("--step") + 1] for job in jobs] == ["fit", "hold_fit", "causal", "hold_causal",
                                                                             "compare"]
    assert all(job["name"].endswith("digit_cmd_20261001") for job in jobs)
    assert all("consequence_from_command" in job["why"] for job in jobs[:2])
    assert all("encoded" in job["why"] for job in jobs[2:])
