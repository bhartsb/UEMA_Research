# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The saved-checkpoint proprioception encoder check (docs/gain_of_function_v4/proprio_encoder_check_2026-10-11/
encoder_check.py): its planted cases, its refusals and its declaration. Light: synthetic rows and small random encoders,
no world, no checkpoint, no graphics card."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

TOOL = Path(__file__).resolve().parents[4] / "docs/gain_of_function_v4/proprio_encoder_check_2026-10-11/encoder_check.py"


@pytest.fixture(scope="module")
def ec():
    spec = importlib.util.spec_from_file_location("proprio_encoder_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["planted_identical", "planted_scaled", "planted_gain_curve",
                                  "planted_svd_agreement", "planted_reference_copy", "planted_null",
                                  "planted_cross_path", "planted_missing_tensor", "planted_reproduction",
                                  "planted_heading_only", "planted_alignment", "planted_nondeterministic",
                                  "planted_joint_groups"])
def test_planted_case_holds(ec, case):
    r = getattr(ec, case)()
    assert r["held"], r


def test_validate_runs_every_planted_case(ec):
    assert len(ec.PLANTED) == 13


def test_the_penalty_grid_is_0_and_200_times_powers_of_4(ec):
    assert len(ec.LAMS) == 28 and ec.LAMS[0] == 0.0 and ec.LAMS[ec.LAM_200] == 200.0
    assert abs(ec.LAMS[1] - 200.0 / 4 ** 24) < 1e-20 and ec.LAMS[-1] == 3200.0
    assert all(lam in ec.LAMS for lam in ec.SHARED_LAMS) and len(ec.SHARED_LAMS) == 11


def _rows(seed=5):
    g = torch.Generator().manual_seed(seed)
    Z = torch.cumsum(torch.randn(3, 300, 24, generator=g), dim=1)
    Z = (Z - Z.mean(1, keepdim=True)) / Z.std()
    T = Z[..., :5] @ torch.randn(5, 7, generator=g) + 0.5 * torch.randn(3, 300, 7, generator=g)
    return Z, T


def test_the_float64_curve_equals_posture_contents_algorithm_in_float64_at_all_11_penalties(ec):
    """Part (a), the gate: the solve keeps posture_content's definition at every shared penalty within 5.0e-4."""
    Z, T = _rows()
    S = ec.FoldSolver(Z).scores(T)
    for lam in ec.SHARED_LAMS:
        ref = ec.posture_content_reference(Z, T, lam)
        assert float((S[ec.LAMS.index(lam)] - ref).abs().max()) <= ec.AGREE_TOL, lam
    r = ec.agreement(Z, T)
    assert r["held"] and r["gate"]["held"] and len(r["gate"]["by_penalty"]) == 11


def test_the_float32_comparison_is_recorded_at_all_11_penalties_and_never_gates(ec):
    """Part (b): the difference from production's float32 posture_content is recorded per penalty with its label, and
    a float32 difference over the tolerance does not change `held` (only part (a) gates)."""
    from mimo_bridge import proprio_token as PT
    Z, T = _rows(6)
    r = ec.agreement(Z, T)
    rec = r["float32_recorded"]
    assert rec["gated"] is False and len(rec["by_penalty"]) == 11
    assert rec["labels"] == {"%.5g" % ec.SHARED_LAMS[0]: "production's float32 Cholesky rounding"}
    S = ec.FoldSolver(Z).scores(T)
    for lam in ec.SHARED_LAMS:
        d = float((S[ec.LAMS.index(lam)] - PT.posture_content(Z, T, lam=lam).double()).abs().max())
        assert abs(rec["by_penalty"]["%.5g" % lam] - float("%.3g" % d)) <= 1e-12
    over = dict(r, float32_recorded=dict(rec, by_penalty={k: 1.0 for k in rec["by_penalty"]}))
    s = ec.agreement_summary([over])
    assert s["gate_held"] and s["float32_gated"] is False and len(s["float32_over_tolerance_at"]) == 11


def test_the_reference_copy_reproduces_production_in_float32(ec):
    from mimo_bridge import proprio_token as PT
    Z, T = _rows(7)
    for lam in ec.SHARED_LAMS:
        assert torch.equal(ec.posture_content_reference(Z, T, lam, dtype=torch.float32), PT.posture_content(Z, T, lam=lam))


def test_the_rule_reads_unreadable_when_a_best_sits_at_a_grid_end(ec):
    rows = [dict(window="w", part="sub %d of 4" % i, curve_best=[0.5, 0.4, -0.1], curve_best_at_end=[None, None])
            for i in range(1, 8)]
    assert ec.curve_rule(rows)["reading"] == "later encoder lower"
    rows[3]["curve_best_at_end"] = ["low", None]
    r = ec.curve_rule(rows)
    assert r["reading"] == "unreadable: best at the grid's end" and r["parts_with_best_at_grid_end"]


def test_build_registry_refuses_a_missing_tensor(ec):
    sd = {k: v.clone() for k, v in ec._random_registry(1).state_dict().items()}
    sd.pop("encoders.proprioception.mlp.1.bias")
    with pytest.raises(RuntimeError):
        ec.build_registry(sd)


def test_a_constant_target_reads_no_score(ec):
    """A score whose target did not move is None, never the score of 1 a constant gives (content_curve)."""
    B, n = 2, 120
    Z = torch.randn(B, n, 8)
    c = ec.content_curve(Z, torch.zeros(B, n, 5))
    assert c["best"] == [None, None] and all(v == [None, None] for v in c["values"])


def test_the_rule_reads_lower_higher_within_and_too_few(ec):
    assert ec.rule_reading([-0.1, -0.12, -0.09, -0.11, -0.1, -0.13, -0.1])["reading"] == "later encoder lower"
    assert ec.rule_reading([0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 0.1])["reading"] == "later encoder higher"
    assert ec.rule_reading([-0.1, 0.12, -0.09, 0.11, -0.1, 0.13, 0.1])["reading"] == "within the spread across parts"
    assert ec.rule_reading([-0.1, -0.1, -0.1])["reading"] == "too few parts"
    # lower on 6 of 7 parts but one large rise: the bound crosses 0, so within the spread
    assert ec.rule_reading([-0.01, -0.02, -0.01, -0.02, -0.01, -0.02, 0.5])["reading"] == \
        "within the spread across parts"


def test_run_refuses_without_end_unless_dev_and_writes_nothing_else(ec, tmp_path):
    out = tmp_path / "o"
    assert ec.main(["run", "--start", str(tmp_path / "missing"), "--out", str(out)]) == 2
    assert not (out / "status.json").exists()
    assert ec.main(["run", "--start", "a", "--end", "b", "--end-preserved", "c", "--out", str(out)]) == 2


def test_end_preserved_refuses_a_manifest_without_pass(ec, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "MANIFEST.json").write_text(json.dumps({"verdict": "fail", "step": 489350}), encoding="utf-8")
    out = tmp_path / "o"
    assert ec.main(["run", "--start", str(tmp_path / "missing"), "--end-preserved", str(dest), "--out", str(out)]) == 1
    s = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert s["verdict"] == "invalid" and any("not pass" in r for r in s["invalid_reasons"])


def test_a_dev_run_on_a_missing_checkpoint_is_invalid_with_its_reason(ec, tmp_path):
    out = tmp_path / "o"
    code = ec.main(["run", "--start", str(tmp_path / "missing"), "--out", str(out), "--dev"])
    s = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert code == 1 and s["verdict"] == "invalid" and s["mode"] == "dev"
    assert any("start checkpoint" in r for r in s["invalid_reasons"])
    d = json.loads((out / "declaration.json").read_text(encoding="utf-8"))
    for key in ("choices", "score_definitions", "targets", "not_done", "question_1_rule"):
        assert key in d
    assert d["question_1_rule"]["score"].startswith("curve_best")
    assert ec.main(["run", "--start", str(tmp_path / "missing"), "--out", str(out), "--dev"]) == 2, \
        "an existing status.json is never overwritten without --replace"


def test_the_window_parts_cover_the_window(ec):
    parts = ec.window_parts(1200, 4)
    assert [(a, z) for _p, a, z in parts] == [(0, 300), (300, 600), (600, 900), (900, 1200)]
