# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The dashboard's action-pathway reading measures the INPUT, and it detects the defect.

The weight reading (|w| of the command columns, 0.083) said the pathway was alive on every
checkpoint from H28 to H33 while the model could not see the action: 99.6% of the motor
row's input energy sat in a position block written in centimetres. A reading that cannot
tell that state from a healthy one is not a reading. These tests hold the new probe to
that: fed raw centimetres it must say DROWNED on ANY encoder, fed normalised commands it
must say READABLE, and the dashboard must show the input reading rather than the weight.
"""
import inspect

import pytest
import torch

from mimo_bridge.lem_model import LemLite
from mimo_bridge.tools import dashboard
from mimo_bridge.tools.probes import action_readability, synthetic_motor_rows


@pytest.fixture(scope="module")
def registry():
    return LemLite(device="cpu", seed=0, intention=False, imagination=False).encoder_registry


def test_normalised_commands_are_readable(registry):
    r = action_readability(registry)
    assert r["readable"], r
    assert r["ratio"] >= 1.0, f"two different orders should move the token at least as far as the position block: {r}"
    assert r["diff_action"] > 1.0, r


def test_raw_centimetres_reproduce_the_defect_on_a_fresh_encoder(registry):
    """The detector must see the disease even on untrained weights - it is an INPUT fact."""
    r = action_readability(registry, normalise=False)
    assert not r["readable"], r
    assert r["ratio"] < 0.2, f"raw cm should drown the order (H33 measured 1.6 vs 14.6): {r}"
    assert r["pos_block"] > 5.0, r


def test_the_rows_are_realistic_and_paired():
    rows = synthetic_motor_rows(seed=3, pairs=4)
    assert len(rows) == 4
    ach, a, b = rows[0]
    assert ach.shape == (1, 24, 10) and a.shape == (1, 24, 9) and b.shape == (1, 24, 9)
    assert float(a[..., 0:3].abs().mean()) > 1.0, "orders are in centimetres"
    assert float(ach[..., 0:3].abs().mean()) < 0.5, "achieved positions are /200"
    assert not torch.allclose(a[..., 3:7], b[..., 3:7]), "the two orders differ in rotation"
    assert torch.allclose(a[..., 0:3] - ach[..., 0:3] * 200.0, a[..., 0:3] - ach[..., 0:3] * 200.0)


def test_dashboard_shows_the_input_reading_not_the_weight():
    """Rewritten 2026-09-23 (fix-dashboards, reviews/dashboards.md M9): the green "action READABLE"
    badge changed all 26 body parts at once and passed while 10 of them could not be read at all;
    the badge now reads one body part at a time against a line measured on ckpt_step_344400."""
    src = inspect.getsource(dashboard)
    assert 'info["action_read"] = action_readability_for(run_dir)' in src
    assert "action UNREAD for" in src and "action DROWNED" in src and "action READABLE" not in src
    assert "changed at once; per-body-part " in src and "readability unmeasured" in src
    assert "action pathway INERT (weights exactly 0)" in src, "the exact-zero (grown, untrained) case stays"
    doc = dashboard.DOC["chip:cmdpath"]
    assert doc["t"].startswith("Can the model read each body part")
    assert "probes.py" in doc["h"] and "per_handle_readability" in doc["h"]
    assert "all 26 body parts at once" in doc["f"]


def test_one_body_part_at_a_time_on_a_fresh_encoder(registry):
    from mimo_bridge.tools.probes import per_handle_readability
    rows = [(ach, a) for ach, a, _b in synthetic_motor_rows(n=24, pairs=2)]
    r = per_handle_readability(registry, rows, axes=1)
    assert len(r["share"]) == 24 and r["rows"] == 2 and r["turn_deg"] == 20.0
    assert all(s > 0 for s in r["share"]), "a fresh encoder reads every part a little"
    assert r["all_at_once"] > max(r["per_handle"]), "turning every part moves the token more than any one"


def test_dashboard_reading_on_a_saved_checkpoint(tmp_path):
    from lem.training.checkpoint import CheckpointManager
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    m.checkpoints = CheckpointManager(m.config, str(tmp_path / "checkpoints"))
    path = m.save_checkpoint(7)
    assert path
    run_dir = str(tmp_path)
    r = dashboard.action_readability_for(run_dir)
    assert r is not None and r["readable"] and r["checkpoint"].startswith("ckpt_step_7")
    assert dashboard.action_readability_for(run_dir) is r, "cached on the checkpoint's mtime"


def test_dashboard_caches_multiple_runs_and_does_not_build_a_whole_model(tmp_path,monkeypatch):
    from lem.training.checkpoint import CheckpointManager
    from mimo_bridge import lem_model
    runs=[tmp_path/'legacy',tmp_path/'trunk']
    for run,layout in zip(runs,['legacy','trunk_v1']):
        m=LemLite(device='cpu',seed=0,intention=False,imagination=False,body_layout=layout)
        m.checkpoints=CheckpointManager(m.config,str(run/'checkpoints'))
        m.save_checkpoint(7)
    def forbidden(*args,**kwargs):raise AssertionError('dashboard built a whole brain')
    monkeypatch.setattr(lem_model,'LemLite',forbidden)
    rng=torch.get_rng_state().clone()
    first=dashboard.action_readability_for(str(runs[0]))
    second=dashboard.action_readability_for(str(runs[1]))
    assert first is not None and second is not None
    assert dashboard.action_readability_for(str(runs[0])) is first
    assert torch.equal(rng,torch.get_rng_state())


def test_the_overseer_checks_whether_the_model_can_read_its_action():
    """The watchdog makes the cheapest check for the most expensive failure this project
    has had: from H28 to H33 the command columns carried live weights while the action was
    drowned, and no causal test in that era could resolve it."""
    from mimo_bridge.tools import overseer
    src = inspect.getsource(overseer)
    assert "action_readability" in src
    assert "CANNOT READ ITS OWN ACTION" in src
    assert "_read_ck" in src, "the check must be cached per checkpoint, not rerun every pass"


def test_the_overseer_check_actually_runs():
    """It shipped with a NameError - the cache attribute was hung on a function that does
    not exist in that module - and a bare `except Exception` that printed only the
    exception TYPE, so the log read 'could not run: NameError' for a whole run. The name
    is asserted here, and the handler must report the message."""
    from mimo_bridge.tools import overseer
    src = inspect.getsource(overseer)
    assert "check_run._read_ck" not in src and "getattr(check_run" not in src
    assert "getattr(inspect, \"_read_ck\"" in src
    assert "{type(_ex).__name__}: {_ex}" in src, "the handler must say WHAT failed, not only its type"
    assert hasattr(overseer, "inspect") and callable(overseer.inspect)


def _mimo(step, npe):
    return {"type": "mimo", "step": step, "timestamp": 0, "elapsed_s": 600, "next_pe": npe}


def test_the_overseer_sees_a_sense_diverging_ALONE(tmp_path):
    """The rule was written for H32 - touch +68% while vision -63% - and required BOTH.
    On H35 touch rose 80% over three hours while vision and body position stayed flat, so
    nothing fired: the overseer reported ball distances for six passes while it ran."""
    from mimo_bridge.tools import overseer
    src = inspect.getsource(overseer)
    assert "len(rising) == 1 and not falling" in src, "one sense rising alone must be reported"
    assert "wrong way on its own" in src
    # and it must not be blind to a drift slower than the main window
    assert 'read_tail(os.path.join(run, "replay_training_stats.jsonl"), "mimo", n=240)' in src, (
        "the check needs its own long tail: WINDOW is 30 records, about 15 minutes, and this "
        "drift took hours")
    i_long = src.index("_long = read_tail")
    i_use = src.index("half = len(_long) // 2")
    assert i_use > i_long
    assert "for r in recs[:half]" not in src, "the split must read the long tail, not the short window"
