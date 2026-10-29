# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import os

import numpy as np
import pytest

from mimo_bridge.latent_body_clone import compare


class Session:
    n_handles = 2
    def __init__(self):
        self.position = 12
        self.last_motor_substeps = None
        self.last_hand_joint_goal = None
        self.starts = []
    def snapshot(self): return {"position": self.position}
    def restore(self, state): self.position = state["position"]
    def run_arm(self, state, command, steps):
        self.restore(state); self.starts.append(self.position)
        self.position += float(command[0, 0, 0]) * steps
        self.last_motor_substeps = command.copy()
        return [{"pos_cm": np.full((2, 3), self.position)}]


def inputs():
    orders = {name: np.full((1, 2, 9), value, np.float32)
              for name, value in (("direct", 1), ("synergy", 2), ("latent", 3))}
    return dict(candidates=orders, score=lambda row: float(row["pos_cm"].mean()), steps=1,
                predicted_scores=dict(direct=1, synergy=2, latent=3), repeat_noise=0.,
                candidate_counts=dict(direct=5, synergy=5, latent=5), group="held_out_body", quality="plumbing")


def test_all_arms_repeat_from_the_identical_start_and_caller_is_restored():
    session = Session(); before = copy.deepcopy(inputs()["candidates"])
    result = compare(session, **inputs())
    assert session.starts == [12] * 6 and session.position == 12
    assert result["ranking"]["correct_pairs"] == 3
    assert not result["production_control"]
    assert all(np.array_equal(before[k], inputs()["candidates"][k]) for k in before)


def test_planted_replay_failure_is_refused_and_caller_restored():
    session = Session(); real = session.run_arm
    def faulty(*args, **kwargs):
        values = real(*args, **kwargs)
        values[-1]["pos_cm"] += len(session.starts)
        return values
    session.run_arm = faulty
    with pytest.raises(ValueError, match="replay differs"):
        compare(session, **inputs())
    assert session.position == 12


@pytest.mark.slow
def test_real_saved_world_replays_the_clone_harness(tmp_path, monkeypatch):
    checkpoint = os.environ.get("UEMA_LATENT_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("C1 supplies the detached qualification checkpoint")
    from mimo_bridge.tools import gof_probe_harness as H
    monkeypatch.setattr(H, "PROBES_ROOT", tmp_path)
    frozen = H.load_frozen(checkpoint)
    episode, reason = H.load_episode(frozen)
    assert episode is not None, reason
    session = H.Session(frozen, episode=episode, body=0)
    try:
        session.observe()
        command = session.hold_order()
        result = compare(session, {name: command.copy() for name in ("direct", "synergy", "latent")},
            lambda row: float(np.square(row["pos_cm"]).mean()), steps=1,
            predicted_scores=dict(direct=0., synergy=0., latent=0.), repeat_noise=0.,
            candidate_counts=dict(direct=1, synergy=1, latent=1), group="plumbing_only", quality="plumbing")
        assert all(row["repeated_exactly"] for row in result["candidates"].values())
    finally:
        session.close()
