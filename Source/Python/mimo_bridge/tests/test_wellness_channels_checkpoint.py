# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The wellness channels on the real H84 checkpoint (DESIGN.md D2, D3; lane-b3-wellness).

Loading ckpt_step_344400 with every wellness-channel flag on reinitialises NOTHING: the same
parameters, every weight and every saved Adam moment equal to the flags-off load. With the deploy's
fresh networks on (the vestibular encoder, the efference heads) the load leaves exactly their
learning-progress keys out for LP_HOLDOUT_STEPS model steps, and that hold-out survives a save.
CPU. The checkpoint is found in this tree, the main tree (a worktree holds no checkpoints) or
wherever the archiver moved it (tests/saved_checkpoints.py); UEMA_H84_CKPT still names any other
place. Skipped only where it is nowhere."""
import gc
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from mimo_bridge.tests.saved_checkpoints import h84_344400

H84 = h84_344400()
CKPT = H84.path

pytestmark = [pytest.mark.slow, H84.mark()]

ON = {"controllability": "on", "learning_progress": "on", "wellness_rank": "normalized",
      "curiosity_mode": "additive", "consequence_self_contact": "exclude"}


def _build(out, **flags):
    from mimo_bridge.tools.self_recovery_experiment import configuration
    from mimo_bridge import train_mimo as TM
    H84.require()
    args = configuration(CKPT, out)
    args.caregiver_record_dir = ""
    for k, v in flags.items():
        setattr(args, k, v)
    m = TM.build_learner(args, None)
    res = m.load_checkpoint(str(CKPT), load_optimizer=True)
    return m, res


def test_the_flags_reinitialise_nothing_and_keep_every_adam_moment(tmp_path):
    torch.set_num_threads(4)
    off, r_off = _build(tmp_path)
    on, r_on = _build(tmp_path, **ON)
    assert r_off["reinitialized_components"] == [] and r_on["reinitialized_components"] == []
    assert len(on.params) == len(off.params)
    same = 0
    for a, b in zip(off.params, on.params):
        assert torch.equal(a, b)
        sa, sb = off.optimizer.state.get(a), on.optimizer.state.get(b)
        assert (not sa) == (not sb)
        if sa:
            assert torch.equal(sa["exp_avg"], sb["exp_avg"])
            assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])
            same += 1
    assert same > 0
    assert off.wellness_channels is None and "wellness_channels" not in off.receipt
    rec = on.receipt["wellness_channels"]
    assert rec["version"] == "wellness_channels_v2" and rec["gate_reads"] == "harm"
    assert rec["detrend_steps"] is not None and rec["control_kind"] == "shuffled order"
    # nothing was started fresh, so nothing sits out of learning progress
    assert on.wellness_channels.lp.holdout == {}


def test_the_deploys_fresh_networks_sit_out_of_learning_progress_and_the_holdout_is_saved(tmp_path):
    from lem.model import wellness_channels as WC
    from lem.training.checkpoint import CheckpointManager
    torch.set_num_threads(4)
    m, res = _build(tmp_path, learning_progress="on", vestibular_sense="on", efference_heads="on")
    assert "efference_heads" in res["reinitialized_components"]
    lp = m.wellness_channels.lp
    assert lp.holdout == {"efference_body": WC.LP_HOLDOUT_STEPS,
                          "efference_gaze": WC.LP_HOLDOUT_STEPS,
                          "vestibular": WC.LP_HOLDOUT_STEPS}
    for _ in range(3):
        lp.tick_holdout()
    tmp = Path(tempfile.mkdtemp(prefix="lb3_ckpt_"))
    try:
        m.checkpoints = CheckpointManager(m.config, str(tmp))
        p = m.save_checkpoint(344400)
        rt = torch.load(os.path.join(p, "runtime_state.pt"), weights_only=False)
        assert rt["wellness_channels"]["learning_progress"]["holdout"] == {
            k: WC.LP_HOLDOUT_STEPS - 3 for k in ("efference_body", "efference_gaze", "vestibular")}
        del m, lp, rt
        gc.collect()
        m2, _ = _build(tmp_path, learning_progress="on", vestibular_sense="on", efference_heads="on")
        r2 = m2.load_checkpoint(p, load_optimizer=True)
        # the saved checkpoint has the heads and the vestibular encoder: nothing is fresh at this
        # load, and the saved hold-out continues where it stopped
        assert "efference_heads" not in (r2.get("reinitialized_components") or [])
        assert m2.wellness_channels.lp.holdout == {
            k: WC.LP_HOLDOUT_STEPS - 3 for k in ("efference_body", "efference_gaze", "vestibular")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
