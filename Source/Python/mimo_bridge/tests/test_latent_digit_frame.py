# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The index-level repair of Sep 30, 2026 (docs/gain_of_function_v4/agents/latent_index_2026-09-30.md).

digit_frame "wrist" expresses every digit row relative to its own wrist, and digit_scales scale digit rows on their
own. Each test names the defect it catches; each was run with that defect planted and failed.
"""
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from lem.latent_body.model import LocalPrediction
from lem.latent_body.trajectory import (action_rows, change_rows, digit_wrists, end_positions, make_batch,
                                        row_scales)
from lem.planner.latent_rollout import handle_targets
from mimo_bridge.tests.latent_body_fixtures import config, example, tree

WRIST = dict(hand_frame="actual_wrist", digit_frame="wrist")


def _unit(rows, seed):
    return torch.nn.functional.normalize(torch.randn(rows, 4, generator=torch.Generator().manual_seed(seed)), dim=-1)


def _example(seed=0):
    """A body whose bones all have distinct orientations, positions and velocities, and a wrist-relative command."""
    g = torch.Generator().manual_seed(seed)
    ex = example()
    ex.action_contract = "wrist_relative_joint_v1"
    for k, o in enumerate((ex.history, ex.start, ex.end)):
        o.bones[:, :4] = _unit(26, seed * 10 + k)
        o.bones[:, 4:7] = torch.randn(26, 3, generator=g) * 10
        o.bones[:, 7:10] = torch.randn(26, 3, generator=g) * 20
    ex.executed[:, 3:7] = _unit(26, seed * 10 + 5)
    ex.executed[:, :3] = ex.start.bones[:, 4:7] + torch.randn(26, 3, generator=g) * 2
    return ex


def _carry(ex, side, rot, shift):
    """The end observation = the start with side's wrist and digits carried rigidly by (rot about the wrist, shift)."""
    names = tree().handle_names
    w = names.index("wrist_" + side)
    ids = [w] + [i for i, ww in digit_wrists(names).items() if ww == w]
    end = ex.end
    end.bones = ex.start.bones.clone()
    pivot = ex.start.bones[w, 4:7].double().numpy()
    for i in ids:
        p = ex.start.bones[i, 4:7].double().numpy()
        end.bones[i, 4:7] = torch.tensor(rot.apply(p - pivot) + pivot + shift, dtype=torch.float32)
        end.bones[i, :4] = torch.tensor((rot * R.from_quat(ex.start.bones[i, :4].double().numpy())).as_quat(),
                                        dtype=torch.float32)
    return ids


def test_a_hand_carried_rigidly_does_not_move_its_digits_in_the_wrist_frame():
    # Defect caught: digit change measured in world coordinates (the refit's index code then learns the arm's
    # carry: it read out the wrist's step with R^2 0.47-0.62 on held-out body 7).
    ex, names = _example(), tree().handle_names
    ids = _carry(ex, "L", R.from_rotvec([0.3, -0.2, 0.5]), np.array([2., -1., 3.]))
    world = change_rows(ex.start, ex.end, ids, names, "world")
    wrist = change_rows(ex.start, ex.end, ids, names, "wrist")
    assert world[1:, :3].norm(dim=-1).min() > 0.5           # in the world, every digit was carried
    assert wrist[1:].abs().max() < 1e-4                      # relative to its wrist, no digit moved
    assert torch.allclose(wrist[0], world[0])                # the wrist row itself stays a world row


def test_wrist_frame_change_is_the_change_of_the_digit_pose_seen_from_the_wrist():
    # Defect caught: relative change taken in world axes or with the start wrist for both ends.
    ex, names = _example(1), tree().handle_names
    i, w = names.index("index_R"), names.index("wrist_R")
    got = change_rows(ex.start, ex.end, [w, i], names, "wrist")[1]

    def seen(o):
        W = R.from_quat(o.bones[w, :4].double().numpy())
        F = R.from_quat(o.bones[i, :4].double().numpy())
        return W.inv().apply((o.bones[i, 4:7] - o.bones[w, 4:7]).double().numpy()), W.inv() * F
    p0, q0 = seen(ex.start)
    p1, q1 = seen(ex.end)
    assert np.allclose(got[:3].numpy(), p1 - p0, atol=1e-3)
    assert np.allclose(got[3:].numpy(), (q1 * q0.inv()).as_rotvec(), atol=1e-4)


def test_wrist_frame_actions_reissue_the_recorded_command_with_digit_scales():
    # Defects caught: a digit action not rotated back into the world (missing R_wrist), the digit scale not applied
    # on the way back, and the actual-wrist re-anchoring dropped.
    t, ex = tree(), _example(2)
    digit = dict(state=[2.] * 12, change=[.5] * 6, action=[.7, .8, .9, .3, .4, .5, 1., 1.])
    batch = make_batch(t, [ex], scales=config()["scales"], duration_s=.05, action_contract="wrist_relative_joint_v1",
                       digit_scales=digit, **WRIST)
    predictions = {n.name: LocalPrediction(batch.change[n.name], batch.contact[n.name], batch.action[n.name], {})
                   for n in t.nodes}
    template = ex.executed.clone()
    template[:, :3] = 0
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    out = handle_targets(t, predictions, ex.start, template, config()["scales"], "actual_wrist", "wrist", digit)
    owned = sorted(i for ids in t.owned.values() for i in ids)
    assert torch.allclose(out[owned, :3], ex.executed[owned, :3], atol=2e-3)
    turn = R.from_quat(out[owned, 3:7].double().numpy()) * R.from_quat(ex.executed[owned, 3:7].double().numpy()).inv()
    assert turn.magnitude().max() < 1e-3


def test_end_positions_rebuild_the_world_endpoint_from_wrist_frame_rows():
    # Defect caught: a digit's world endpoint composed without the wrist's own turn (what the clone planner scores).
    t, ex = tree(), _example(3)
    names = t.handle_names
    for root in t.roots:
        ids = list(t.subtree[root])
        for frame in ("world", "wrist"):
            change = change_rows(ex.start, ex.end, ids, names, frame)
            got = end_positions(ex.start, ids, change[None], names, frame)[0]
            assert torch.allclose(got, ex.end.bones[ids, 4:7], atol=2e-3), (root, frame)
    with pytest.raises(ValueError, match="wrist row"):
        i = names.index("index_L")
        end_positions(ex.start, [i], torch.zeros(1, 1, 6), names, "wrist")


def test_parent_and_child_hold_the_same_digit_rows_so_consistency_compares_like_with_like():
    # Defect caught: the switch applied in the child's rows only (cross_level_consistency would then compare a
    # relative finger change with a world one).
    t, ex = tree(), _example(4)
    b = make_batch(t, [ex], scales=config()["scales"], duration_s=.05, action_contract="wrist_relative_joint_v1",
                   **WRIST)
    for side in "LR":
        for parent in ("arm_", "hand_"):
            ids = list(t.subtree[parent + side])
            k = ids.index(t.handle_names.index("index_" + side))
            rows = b.change[parent + side].reshape(len(ids), 6)
            assert torch.allclose(rows[k], b.change["index_" + side][0], atol=1e-6)


def test_digit_scales_apply_to_digit_rows_only():
    # Defect caught: digit scales applied to every row of a part that holds a digit (the wrist and arm included).
    names = tree().handle_names
    ids = list(tree().subtree["hand_L"])
    got = row_scales(names, ids, config()["scales"], "action", dict(state=[2.] * 12, change=[3.] * 6, action=[5.] * 8))
    for k, i in enumerate(ids):
        assert float(got[k, 0]) == (5. if i in digit_wrists(names) else 1.)


def test_default_rows_are_the_earlier_world_rows_and_the_switch_needs_the_actual_wrist():
    # Defect caught: the default path changed (an existing hierarchy would silently change meaning).
    from lem.latent_body.learning import ShadowLearner
    t, ex = tree(), _example(5)
    names = t.handle_names
    ids = list(t.subtree["arm_L"])
    a, b = ex.start.bones[ids], ex.end.bones[ids]
    from lem.latent_body.trajectory import rotation_delta
    assert torch.equal(change_rows(ex.start, ex.end, ids, names),
                       torch.cat((b[:, 4:7] - a[:, 4:7], rotation_delta(a[:, :4], b[:, :4])), -1))
    own = list(t.owned["hand_L"])
    c = ex.executed[own]
    assert torch.equal(action_rows(ex.start, ex.executed, own, names),
                       torch.cat((c[:, :3] - ex.start.bones[own, 4:7],
                                  rotation_delta(ex.start.bones[own, :4], c[:, 3:7]), c[:, 7:9]), -1))
    cfg = dict(config(), action_contract="wrist_relative_joint_v1")
    with pytest.raises(ValueError, match="actual wrist"):
        ShadowLearner(t, dict(cfg, digit_frame="wrist"))
    ShadowLearner(t, dict(cfg, hand_action_frame="actual_wrist", digit_frame="wrist",
                          digit_scales=dict(state=[1.] * 12, change=[1.] * 6, action=[1.] * 8)))


def test_the_causal_test_reads_a_carried_finger_as_still_in_the_wrist_frame():
    # Defect caught: the causal-control tool grading wrist-frame predictions against world movement.
    from mimo_bridge.tools.latent_body_intervention import actual_change
    t, ex = tree(), _example(6)
    ids = _carry(ex, "R", R.from_rotvec([0., 0.4, 0.]), np.array([0., 1., 0.]))
    i = t.handle_names.index("index_R")
    assert actual_change(t, ex.start, ex.end, [i], "wrist").abs().max() < 1e-4
    assert actual_change(t, ex.start, ex.end, [i])[:, :3].norm() > 0.3


def test_split_measures_digit_scales_apart_and_each_group_scores_one_when_still(tmp_path):
    # Defect caught: digit scales measured over every row (the finger rows would stay 45x quieter than the arm's).
    from mimo_bridge.latent_body_runtime import example_record, jsonable
    from mimo_bridge.tools import latent_body_split
    from mimo_bridge.tools.latent_body import read_examples
    rows = []
    for body in (0, 1):
        for step in range(1, 30):
            ex = _example(step + 40 * body)
            for o in (ex.history, ex.start, ex.end):
                o.env_id, o.episode_id = body, 0
            ex.history.frame, ex.start.frame, ex.end.frame = step - 1, step, step + 1
            ex.history.sim_time, ex.start.sim_time, ex.end.sim_time = (step - 1) * .05, step * .05, (step + 1) * .05
            rows.append(example_record(ex))
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r, default=jsonable) + "\n" for r in rows), encoding="utf-8")
    base = dict(config(), action_contract="wrist_relative_joint_v1")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    latent_body_split.main(["--trajectory", str(tmp_path / "t.jsonl"), "--holdout-body", "1", "--base-config",
                            str(tmp_path / "base.json"), "--output", str(tmp_path / "s"), "--scale-rows", "29",
                            "--digit-frame", "wrist", "--digit-scales"])
    cfg = json.loads((tmp_path / "s" / "config.json").read_text(encoding="utf-8"))
    assert cfg["digit_frame"] == "wrist" and set(cfg["digit_scales"]) == {"state", "change", "action"}
    t = tree()
    b = make_batch(t, list(read_examples(tmp_path / "s" / "train.jsonl")), scales=cfg["scales"], duration_s=.05,
                   action_contract="wrist_relative_joint_v1", digit_scales=cfg["digit_scales"], **WRIST)
    digits = digit_wrists(t.handle_names)
    for key, dim, which in (("change", 6, t.subtree), ("action", 8, t.owned)):
        for group in (True, False):
            values = torch.cat([v.reshape(len(v), -1, dim)[:, [h in digits for h in which[n]] if group
                                else [h not in digits for h in which[n]]].reshape(-1, dim)
                                for n, v in getattr(b, key).items()])
            ms = values.square().mean(0)
            assert torch.allclose(ms[ms > 0], torch.tensor(1.), atol=1e-3), (key, group)
