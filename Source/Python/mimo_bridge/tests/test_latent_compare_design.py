# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The hold horizon (trajectory.hold_example, HoldRuns, horizon_steps), the redesigned comparison goals (goal design
recorded_v2: recorded orders of the same body, transplanted, admitted by their separation from the controller's
endpoint) and the training-exposure record (Oct 1, 2026). CPU only, no world. Each test names the defects it catches."""
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from lem.latent_body.learning import ShadowLearner, validate_config
from lem.latent_body.trajectory import Example, HoldRuns, Observation, hold_example, make_batch, row_options
from lem.latent_body.tree import arm_hand_tree, full_body_tree
from mimo_bridge.tests.test_latent_full_body_rows import CONTRACT, NAMES, _obs, config
from mimo_bridge.tools import latent_body_compare as LC

GAZE = NAMES.index("gaze")


def _recording(steps, body=0, orders=None, seed=0):
    """One-step Examples of one body: observation k is the end of row k-1 and the start of row k; `orders[k]` the
    order of row k (default: a new order every 5 rows); every row's eye row differs (each step's mean eye command)."""
    obs = [_obs(k, seed + k) for k in range(steps + 2)]
    for o in obs:
        o.env_id = body
    rows = []
    for k in range(steps):
        order = torch.zeros(26, 9)
        key = (k // 5) if orders is None else orders[k]
        order[:, :3] = float(key)
        order[:, 3:7] = torch.tensor(R.from_rotvec([0.1 * key, 0, 0]).as_quat(), dtype=torch.float32)
        order[:, 7:9] = torch.tensor([.5, .7])
        order[GAZE, 3:7] = torch.tensor(R.from_rotvec([0, 0.02 * k, 0]).as_quat(), dtype=torch.float32)
        order[GAZE, :3] = float(k)
        rows.append(Example(obs[k], obs[k + 1], obs[k + 2], order, CONTRACT))
    return rows


# ------------------------------------------------------------------ the hold horizon
def test_hold_examples_span_one_order_and_average_the_eye_row():
    # Defects caught: the end taken from the first row (a one-step example called a hold); the history moved to the
    # hold's start; the eye row taken from one step; the order compared on the eye row (no run would ever close).
    rows = _recording(10)
    ex = hold_example(rows[:5], NAMES)
    assert ex.start is rows[0].start and ex.history is rows[0].history and ex.end is rows[4].end
    assert ex.end.frame - ex.start.frame == 5 and ex.start.frame - ex.history.frame == 1
    keep = [i for i in range(26) if i != GAZE]
    assert torch.equal(ex.executed[keep], rows[0].executed[keep])
    expected = R.from_quat(np.stack([r.executed[GAZE, 3:7].numpy() for r in rows[:5]])).mean().as_quat()
    got = ex.executed[GAZE, 3:7].numpy()
    assert min(np.abs(got - expected).max(), np.abs(got + expected).max()) < 1e-3
    assert float(ex.executed[GAZE, 0]) == pytest.approx(2.0)          # mean of the steps' eye positions 0..4
    with pytest.raises(ValueError, match="one executed order"):
        hold_example(rows[3:8], NAMES)                                 # crosses an order change


def test_hold_runs_emit_exactly_one_example_per_complete_hold_and_count_cut_runs():
    # Defects caught: a run cut by an order change or a frame gap emitted as a hold; bodies interleaved breaking runs;
    # sliding windows (overlapping examples counted as separate holds).
    a, b = _recording(23, body=0), _recording(23, body=1, seed=40)
    del a[12]                                                          # a frame gap inside body 0's third hold
    runs = HoldRuns(NAMES, 5)
    out = []
    from itertools import zip_longest
    for pair in zip_longest(a, b):
        out += [e for e in (runs.add(x) for x in pair if x is not None) if e is not None]
    lengths = runs.finish()
    by_body = {0: [e for e in out if e.start.env_id == 0], 1: [e for e in out if e.start.env_id == 1]}
    assert [e.start.frame for e in by_body[1]] == [1, 6, 11, 16]       # 23 rows: 4 complete holds, one cut at 3
    assert [e.start.frame for e in by_body[0]] == [1, 6, 16]           # the gap at row 12 cuts the third hold
    assert lengths[1] == {3: 1, 5: 4} and lengths[0] == {2: 2, 3: 1, 5: 3}
    with pytest.raises(ValueError):
        HoldRuns(NAMES, 1)
    # a row whose start is not the previous row's end (the same frame numbers, other bones) breaks the run
    c = _recording(10, body=2)
    c[2] = Example(c[2].history, _obs(3, 99), c[2].end, c[2].executed, CONTRACT)
    c[2].start.env_id = 2
    runs = HoldRuns(NAMES, 5)
    got = [e for e in (runs.add(x) for x in c) if e is not None]
    assert [e.start.frame for e in got] == [6] and runs.breaks[2] == 1
    # a frame gap with identical bones (a still body) breaks it too
    d = _recording(10, body=3)
    jump = Observation(3, 0, d[2].start.frame + 9, d[2].start.sim_time + .45, d[1].end.bones.clone(),
                       d[1].end.contact.clone())
    d[2] = Example(d[2].history, jump, d[2].end, d[2].executed, CONTRACT)
    runs = HoldRuns(NAMES, 5)
    got = [e for e in (runs.add(x) for x in d) if e is not None]
    assert [e.start.frame for e in got] == [6] and runs.breaks[3] == 2


def test_the_hold_horizon_is_declared_validated_and_kept_out_of_the_live_shadow(tmp_path):
    # Defects caught: a hold example accepted by a one-step hierarchy (or a one-step example by a hold hierarchy);
    # the horizon not passed from the configuration to the batch; the live shadow building one-step examples for a
    # hold hierarchy.
    rows = _recording(10)
    hold = hold_example(rows[:5], NAMES)
    tree = full_body_tree(NAMES)
    one, five = ShadowLearner(tree, config()), ShadowLearner(tree, dict(config(), horizon_steps=5))
    one.batch([rows[0]])
    five.batch([hold])
    with pytest.raises(ValueError, match="5 model step"):
        five.batch([rows[0]])
    with pytest.raises(ValueError, match="1 model step"):
        one.batch([hold])
    for bad in (0, 2.5, "5"):
        with pytest.raises(ValueError):
            validate_config(dict(config(), horizon_steps=bad))
    # the change graded over the hold is the end-to-start change over five steps
    batch = five.batch([hold])
    single = make_batch(tree, [Example(hold.history, hold.start, hold.end, hold.executed, CONTRACT)],
                        scales=five.config["scales"], duration_s=.05, action_contract=CONTRACT,
                        hand_frame="actual_wrist", horizon_steps=5, **row_options(five.config))
    assert all(torch.equal(batch.change[k], single.change[k]) for k in batch.change)
    from mimo_bridge.latent_body_runtime import ShadowRuntime
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(dict(config(), horizon_steps=5)), encoding="utf-8")
    args = types.SimpleNamespace(latent_body_output=str(tmp_path / "out"), latent_body_levels="full_body",
                                 latent_body_config=str(path))
    with pytest.raises(ValueError, match="offline only"):
        ShadowRuntime(args, NAMES)


def _split_rows(path, steps=60, bodies=(0, 1)):
    from mimo_bridge.latent_body_runtime import example_record, jsonable
    recs = {b: _recording(steps, body=b, seed=10 * b) for b in bodies}
    with path.open("w", encoding="utf-8") as f:
        for k in range(steps):
            for b in bodies:
                ex = recs[b][k]
                f.write(json.dumps(example_record(ex), default=jsonable) + "\n")
    return recs


def test_the_split_builds_hold_examples_counts_them_and_writes_the_witness_bank(tmp_path):
    # Defects caught: one example per row written under --horizon-steps; the scales measured on one-step rows; the
    # receipt without the examples it yields; no witness bank for the comparison's goals.
    from mimo_bridge.tools import latent_body_split as S
    recs = _split_rows(tmp_path / "t.jsonl")
    base = config()
    base.pop("body_rows"), base.pop("digit_frame"), base.pop("hand_action_frame")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    common = ["--trajectory", str(tmp_path / "t.jsonl"), "--holdout-body", "1", "--base-config",
              str(tmp_path / "base.json"), "--levels", "full_body", "--eye-row-convention", "legacy", "--scale-rows", "8"]
    S.main(common + ["--output", str(tmp_path / "h5"), "--horizon-steps", "5"])
    S.main(common + ["--output", str(tmp_path / "h1")])
    r5 = json.loads((tmp_path / "h5" / "split_receipt.json").read_text(encoding="utf-8"))
    r1 = json.loads((tmp_path / "h1" / "split_receipt.json").read_text(encoding="utf-8"))
    assert r5["examples_by_body"] == {"0": 12, "1": 12} and r5["train_examples"] == 12 and r5["holdout_examples"] == 12
    assert r5["train_rows"] == r1["train_rows"] == 60 and r1["train_examples"] == 60
    assert r5["run_lengths_by_body"]["0"] == {"5": 12}
    cfg = json.loads((tmp_path / "h5" / "config.json").read_text(encoding="utf-8"))
    assert cfg["horizon_steps"] == 5 and "horizon_steps" not in json.loads(
        (tmp_path / "h1" / "config.json").read_text(encoding="utf-8"))
    learner = ShadowLearner(full_body_tree(NAMES), cfg)
    from mimo_bridge.tools.latent_body import read_examples
    held = list(read_examples(tmp_path / "h5" / "holdout.jsonl"))
    learner.batch(held)
    assert all(e.end.frame - e.start.frame == 5 for e in held)
    bank = LC.WitnessBank(tmp_path / "h5" / "witness_bank.pt")
    assert len(bank.env_id) == 24 and set(bank.env_id.tolist()) == {0, 1} and bank.horizon_steps == 5
    assert bank.env_id[0] == 0 and np.allclose(bank.start[0], recs[0][0].start.bones.numpy())
    assert np.allclose(bank.end[0], recs[0][4].end.bones.numpy()) and bank.frame[0] == recs[0][0].start.frame


# ------------------------------------------------------------------ scoring the hold
def test_a_hold_hierarchy_is_scored_on_its_own_hold_prediction(monkeypatch):
    # Defects caught: the hold prediction stretched again by the one-step hold factors; a horizon that is neither
    # one step nor the hold accepted.
    import lem.planner.latent_candidates as LCand
    from mimo_bridge.tests.test_latent_full_body_rows import _example
    assert LC.hold_scoring(dict(duration_s=.05, horizon_steps=5), None) == ("own", None)
    assert LC.hold_scoring(dict(duration_s=.05), None) == ("factor", LC.HOLD_FACTOR)
    assert LC.hold_scoring(dict(duration_s=.05), object()) == ("factors", "HOLD_FACTORS")
    with pytest.raises(ValueError):
        LC.hold_scoring(dict(duration_s=.05, horizon_steps=3), None)
    learner = ShadowLearner(full_body_tree(NAMES), dict(config(), horizon_steps=5))
    ex = _example(2)
    ex.history, ex.start = _obs(0, 4), _obs(1, 5)
    seen = {}
    real = LCand.search

    def spy(model, state, history, score, **kw):
        seen["score"] = score
        return real(model, state, history, score, **kw)
    monkeypatch.setattr(LCand, "search", spy)
    target = ex.start.bones[:, 4:7].numpy() + 1.0
    template = torch.zeros(26, 9)
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    _, _, receipt = LC.latent_order(learner, ex.history, ex.start, target, template, candidates=4, radius=.5,
                                    generator=torch.Generator().manual_seed(0))
    assert receipt["hold_scoring"] == "own" and receipt["hold_factor"] == 1.0 and receipt["horizon_steps"] == 5
    # every part predicts the goal's 1 cm per axis over the hold on its own rows (relative and carried rows: no move
    # relative to their parent, so they are carried the same 1 cm): the hold endpoint is the goal, score 0; the one-step
    # factors (3.09-4.24) applied again would score (factor - 1)^2 x 3 per handle
    from lem.latent_body.trajectory import GROUP_OF
    tree = learner.tree
    body = row_options(learner.config)["body"]
    relative = {i for i, n in enumerate(NAMES) if GROUP_OF[n] == "digit" or n in body.positionless}
    scales = learner.config["body_rows"]["part_scales"]
    preds = {}
    for n in tree.nodes:
        rows = torch.zeros(len(tree.subtree[n.name]), 6)
        for k, i in enumerate(tree.subtree[n.name]):
            if i in tree.owned[n.name] and i not in relative:
                rows[k, :3] = 1.0 / torch.tensor(scales[GROUP_OF[NAMES[i]]]["change"][:3])
        preds[n.name] = types.SimpleNamespace(change=rows.flatten().repeat(4, 1))
    got = seen["score"](preds)
    assert torch.allclose(got, torch.zeros(4), atol=1e-4)


# ------------------------------------------------------------------ the recorded_v2 goals
def _pose(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 10, (26, 3)), R.random(26, random_state=seed).as_quat()


def test_finger_goals_are_scored_relative_to_their_own_wrist():
    # Defects caught: an index goal scored in the world (98-99% of a finger's world movement is the arm carrying it);
    # the wrist of the other side used.
    pos, quat = _pose(1)
    turn = R.from_rotvec([0.3, -0.2, 0.5])
    moved_pos, moved_quat = turn.apply(pos) + [5., -3., 2.], (turn * R.from_quat(quat)).as_quat()
    i, w = NAMES.index("index_L"), NAMES.index("wrist_L")
    assert LC.frame_error_cm(pos, quat, moved_pos, moved_quat, [i], "wrist", NAMES) == pytest.approx(0, abs=1e-9)
    assert LC.frame_error_cm(pos, quat, moved_pos, moved_quat, [i], "world", NAMES) > 1
    bent = moved_pos.copy()
    bent[i] += R.from_quat(moved_quat[w]).apply([0.4, 0., 0.])        # the finger 0.4 cm along its wrist's x
    assert LC.frame_error_cm(pos, quat, bent, moved_quat, [i], "wrist", NAMES) == pytest.approx(0.4, abs=1e-9)
    other = moved_quat.copy()
    other[NAMES.index("wrist_R")] = R.random(random_state=9).as_quat()
    assert LC.frame_error_cm(pos, quat, moved_pos, other, [i], "wrist", NAMES) == pytest.approx(0, abs=1e-9)


def test_a_recorded_order_is_carried_to_the_current_pelvis_and_keeps_the_matched_controls():
    # Defects caught: the recorded world order used where the body now lies elsewhere; the eyes, radius or strength
    # taken from the recording instead of the controller's (the planners' MatchedControls).
    rs = np.zeros((26, 11))
    rs[:, :4] = R.random(26, random_state=3).as_quat()
    rs[:, 4:7] = np.random.default_rng(3).normal(0, 10, (26, 3))
    order = np.zeros((26, 9))
    order[:, :3] = rs[:, 4:7] + np.random.default_rng(4).normal(0, 3, (26, 3))
    order[:, 3:7] = R.random(26, random_state=5).as_quat()
    order[:, 7:9] = [.1, .2]
    template = np.zeros((26, 9))
    template[:, 3:7] = R.random(26, random_state=6).as_quat()
    template[:, 7:9] = [.5, .9]
    same = LC.transplant(order, rs, rs, template, NAMES)
    keep = [i for i in range(26) if i != GAZE]
    assert np.allclose(same[keep, :3], order[keep, :3])
    assert np.allclose(np.abs((same[keep, 3:7] * order[keep, 3:7]).sum(-1)), 1)
    assert np.array_equal(same[:, 7:9], template[:, 7:9]) and np.array_equal(same[GAZE], template[GAZE])
    turn = R.from_rotvec([0, 0, np.pi / 2])
    t = NAMES.index("lower_trunk")
    rc = rs.copy()
    rc[:, 4:7] = turn.apply(rs[:, 4:7] - rs[t, 4:7]) + [30., 0., 0.]
    rc[:, :4] = (turn * R.from_quat(rs[:, :4])).as_quat()
    carried = LC.transplant(order, rs, rc, template, NAMES)
    expected = turn.apply(order[:, :3] - rs[t, 4:7]) + [30., 0., 0.]
    assert np.allclose(carried[keep, :3], expected[keep])
    # the order relative to the pelvis is unchanged
    rel = lambda o, b: R.from_quat(b[t, :4]).inv().apply(o[:, :3] - b[t, 4:7])
    assert np.allclose(rel(carried, rc)[keep], rel(order, rs)[keep])


def _bank(tmp_path, n=40):
    rng = np.random.default_rng(0)
    start = np.zeros((n, 26, 11), dtype=np.float32)
    start[:, :, :4] = [0, 0, 0, 1]
    start[:, :, 4:7] = rng.normal(0, 10, (n, 26, 3))
    start[:, :, 7:10] = rng.normal(0, 5, (n, 26, 3))
    t = NAMES.index("lower_trunk")
    start[n // 2:, t, :4] = R.from_rotvec([np.pi / 2, 0, 0]).as_quat()     # half the recordings lie on their side
    torch.save(dict(env_id=torch.tensor([k % 2 for k in range(n)]), episode_id=torch.zeros(n, dtype=torch.long),
                    frame=torch.arange(n) * 5, start=torch.from_numpy(start), end=torch.from_numpy(start + 1),
                    executed=torch.zeros(n, 26, 9), handle_names=list(NAMES), horizon_steps=5, duration_s=.05,
                    holdout_body=1, trajectory=["x"], action_contract=CONTRACT), tmp_path / "bank.pt")
    return start


def test_witness_candidates_are_the_same_bodys_nearest_starts_lying_the_same_way(tmp_path):
    # Defects caught: another body's recording offered; a start lying on its side offered for a body on its back;
    # candidates not in start-similarity order.
    start = _bank(tmp_path)
    bank = LC.WitnessBank(tmp_path / "bank.pt")
    query = start[4].astype(np.float64).copy()
    query[:, 4:7] += 0.01
    got = bank.nearest(0, query, 5)
    assert got[0][0] == 4 and all(bank.env_id[i] == 0 for i, _ in got)
    assert all(i < 20 for i, _ in got)                                     # none of the recordings on their side
    assert len(bank.nearest(0, query, 40)) == 10                           # body 0 lies on its back in 10 of 20
    assert [d for _, d in got] == sorted(d for _, d in got)
    assert bank.nearest(1, query, 3) and all(bank.env_id[i] == 1 for i, _ in bank.nearest(1, query, 3))
    with pytest.raises(ValueError, match="0.25 s"):
        torch.save(dict(torch.load(tmp_path / "bank.pt", weights_only=True), horizon_steps=1), tmp_path / "b1.pt")
        LC.WitnessBank(tmp_path / "b1.pt")


def test_a_goal_is_admitted_only_beyond_the_controllers_endpoint():
    # Defects caught: the most similar recorded start taken although the controller already ends on its endpoint
    # (the anchoring of Sep 30); a candidate admitted below the declared line.
    cands = [dict(separation_cm=dict(wrist_L=0.5, index_L=0.4)), dict(separation_cm=dict(wrist_L=3.0, index_L=0.1)),
             dict(separation_cm=dict(wrist_L=2.5, index_L=0.35))]
    assert LC.choose_witness(cands, "wrist_L") == 1 and LC.choose_witness(cands, "index_L") == 0
    assert LC.choose_witness(cands[:1], "wrist_L") is None
    assert LC.SEPARATION_CM["wrist_L"] > 2 * LC.SEPARATION_CM["index_L"]


def test_the_request_change_says_whether_the_planners_could_issue_the_witness():
    names = NAMES
    base = np.zeros((26, 9))
    base[:, 3:7] = [0, 0, 0, 1]
    near = base.copy()
    near[0, 2] += 1.7
    near[3, 3:7] = R.from_rotvec([0, 0, np.radians(10)]).as_quat()
    far = base.copy()
    far[0, 0] += 22.
    assert LC.request_change(near, base, names)["inside_planner_region"]
    out = LC.request_change(far, base, names)
    assert not out["inside_planner_region"] and out["max_position_cm"] == pytest.approx(22)
    eyes = base.copy()
    eyes[GAZE, :3] = 50.                                                   # the eyes are matched, never counted
    assert LC.request_change(eyes, base, names)["inside_planner_region"]
    # the planners bound the fingers by their joint goals (11 degrees each), not by the finger rows' place
    finger = NAMES.index("index_L")
    moved = base.copy()
    moved[finger, :3] += 3.0
    fingers = [i for i, n in enumerate(NAMES) if n.split("_")[0] in ("thumb", "index", "middle", "ring", "pinky")]
    joints, same = np.zeros(20), np.zeros(20)
    assert LC.request_change(moved, base, names, fingers, joints, same)["inside_planner_region"]
    joints[4] = np.radians(12.)
    out = LC.request_change(moved, base, names, fingers, joints, same)
    assert not out["inside_planner_region"] and out["max_joint_deg"] == pytest.approx(12.)


def test_the_v2_goals_follow_what_the_tree_owns_and_carries():
    assert set(LC.declared_goals(arm_hand_tree(NAMES))) == {"wrist_L", "index_L", "index_R"}
    assert set(LC.declared_goals(full_body_tree(NAMES))) == set(LC.GOALS_V2)
    assert LC.declared_goals(full_body_tree(NAMES))["index_L"] == (("index_L",), "wrist")
    with pytest.raises(ValueError):
        LC.declared_goals(arm_hand_tree(NAMES), "upperarms")
    assert set(LC.declared_goals(full_body_tree(NAMES), design="anchored_v1")) == set(LC.GOALS)


def _rows(primary, direct, synergy, controller, bodies, admitted=None, goal="g"):
    out = []
    for k, (l, d, s, c, b) in enumerate(zip(primary, direct, synergy, controller, bodies)):
        out.append(dict(goal=goal, body=b, admitted=True if admitted is None else admitted[k],
                        outcomes=dict(latent=dict(goal_error_cm=l), direct=dict(goal_error_cm=d),
                                      synergy=dict(goal_error_cm=s), controller=dict(goal_error_cm=c),
                                      witness=dict(goal_error_cm=0.0))))
    return out


def test_the_summary_tests_only_admitted_goals_and_reports_every_arm_against_the_controller():
    # Defects caught: a not-admitted start counted as a comparison; a goal decided on too few admitted starts; the
    # gap closed measured against the planners instead of the controller.
    n = 18
    bodies = [b for b in range(8) for _ in range(2)] + [7, 7]
    admitted = [True] * 13 + [False] * 5
    rows = _rows([1.] * n, [2.] * n, [2.] * n, [4.] * n, bodies, admitted)
    s = LC.summarize(rows, [7])
    g = s["by_goal"]["g"]
    assert g["admitted"] == 13 and g["starts"] == 18 and g["latent_vs_direct"]["starts"] == 13
    assert g["wins_needed"] == 10 and g["better_than_planner"] and s["not_admitted"]["g"] == 5
    assert g["gap_closed_median"]["latent"] == pytest.approx(.75) and g["gap_closed_median"]["witness"] == 1.0
    assert g["beats_controller"]["direct"]["wins"] == 13 and "witness" not in g["beats_controller"]
    stepped = _rows([1.] * n, [2.] * n, [2.] * n, [4.] * n, bodies)
    for r in stepped:
        r["outcomes"]["latent_step"] = dict(goal_error_cm=3.)
    assert LC.summarize(stepped, [7])["by_goal"]["g"]["latent_vs_latent_step"]["wins"] == 18
    # the held-out line: one sign per start, the start's goals pooled (they share one goal pose)
    held = []
    for start, gain in enumerate([1, 1, 1, 1, 1, 1, -1, 1]):
        for goal in ("a", "b", "c"):
            r = _rows([4. - gain] , [4.], [4.], [8.], [7], goal=goal)[0]
            r["warmup"] = start
            held.append(r)
    h = LC.summarize(held, [7])["held_out"]
    assert h["starts"] == 8 and h["comparisons"] == 24 and h["latent_vs_direct"]["wins"] == 7 and h["passes"]
    assert not LC.summarize(held[:12], [7])["held_out"]["passes"]       # 4 starts: undecidable
    few = LC.summarize(_rows([1.] * 9, [2.] * 9, [2.] * 9, [4.] * 9, bodies[:9]), [7])
    assert not few["by_goal"]["g"]["decided"] and not few["by_goal"]["g"]["better_than_planner"]


def test_the_summary_still_applies_the_sign_test_lines():
    n = 18
    bodies = [b for b in range(8) for _ in range(2)] + [7, 7]
    for wins, passes in ((13, True), (12, False)):
        latent = [1.] * wins + [3.] * (n - wins)
        assert LC.summarize(_rows(latent, [2.] * n, [2.] * n, [2.] * n, bodies), [7])["by_goal"]["g"][
            "better_than_planner"] is passes


# ------------------------------------------------------------------ training exposure
def _fit_folder(tmp_path, train_examples, updates, examples, horizon=1):
    split, fit = tmp_path / "split", tmp_path / "fit"
    split.mkdir()
    fit.mkdir()
    (split / "split_receipt.json").write_text(json.dumps(dict(train_rows=152474, train_examples=train_examples)),
                                              encoding="utf-8")
    (fit / "receipt.json").write_text(json.dumps(dict(
        data=dict(path=str(split / "train.jsonl")), order=dict(epochs=2), training_body_episodes=[[0, 1], [1, 1]],
        held_out_examples=10, held_out_hindsight_losses=dict(future=.2), held_out_future_if_nothing_moves=1.)),
        encoding="utf-8")
    learner = types.SimpleNamespace(updates=updates, examples=examples, config=dict(horizon_steps=horizon))
    return fit / "hierarchy.pt", learner


def test_every_comparison_records_both_sides_training_exposure(tmp_path):
    # Defects caught: a comparison reported without what each side learned from (the Sep 30 report: 304,896 example
    # presentations against 3,578,800 live body-moments, never stated); presentations of five-step examples counted
    # as one body-moment each.
    path, learner = _fit_folder(tmp_path, 30464, 4760, 304640, horizon=5)
    e = LC.hierarchy_exposure(path, learner)
    assert e["distinct_training_examples"] == 30464 and e["distinct_body_moments"] == 152320
    assert e["body_moment_presentations"] == 304640 * 5 and e["training_bodies"] == [0, 1]
    frozen = types.SimpleNamespace(load=dict(step=447350), info=dict(checkpoint="ck", lineage_step=447350),
                                   args=types.SimpleNamespace(n_envs=8, replay="on", replay_every=2, replay_batch=24))
    p = LC.production_exposure(frozen)
    assert p["live_body_moments"] == 3578800
    exposure = dict(production=p, hierarchies=dict(latent=e))
    ratios = LC.exposure_ratios(exposure)
    assert ratios["latent"]["production_over_distinct"] == pytest.approx(3578800 / 152320)
    record = dict(latent_arms=dict(latent={}), training_exposure=exposure)
    assert LC.check_exposure(record) is record
    for broken in (dict(latent_arms=dict(latent={})), dict(latent_arms=dict(latent={}, latent_step={}),
                                                           training_exposure=exposure)):
        with pytest.raises(ValueError, match="training exposure"):
            LC.check_exposure(broken)
    (tmp_path / "fit" / "receipt.json").unlink()
    with pytest.raises(ValueError, match="training exposure"):
        LC.hierarchy_exposure(path, learner)


# ------------------------------------------------------------------ the causal test over the hold
def test_the_causal_test_holds_each_order_for_the_hierarchys_horizon(monkeypatch):
    # Defects caught: a hold hierarchy's codes executed for one model step and graded against its hold prediction;
    # the one-step test changed; the hold execution not checked against the observed path.
    import mimo_bridge.latent_body_runtime as RT
    from mimo_bridge.tests.test_latent_full_body_tools import _Session
    from mimo_bridge.tools import latent_body_intervention as LI
    monkeypatch.setattr(RT, "observed_contract", lambda frame: CONTRACT)

    class HoldSession(_Session):
        def __init__(self):
            super().__init__()
            self.calls = []

        def run_arm(self, snap, order, steps, gaze_rad=None):
            self.calls.append(("run_arm", steps))
            super().run_arm(snap, order, steps, gaze_rad=gaze_rad)

        def physics_only(self, snap, order, steps, gaze_rad=None, render=False):
            assert render, "the hold runs without the rendered frames run_arm has (the eye reflex observes them)"
            self.calls.append(("physics_only", steps))
            _Session.run_arm(self, snap, order, steps, gaze_rad=gaze_rad)
    args = types.SimpleNamespace(warmup_steps=2, codes=4, centre="mean", seed=0)
    for horizon, kinds in ((5, {("physics_only", 5), ("run_arm", 5)}), (1, {("run_arm", 1)})):
        cfg = dict(config(), horizon_steps=horizon) if horizon > 1 else config()
        learner = ShadowLearner(full_body_tree(NAMES), cfg)
        session = HoldSession()
        centres = {n.name: torch.zeros(1, learner.model.latent_dim) for n in learner.tree.nodes}
        spread = {n.name: torch.ones(learner.model.latent_dim) for n in learner.tree.nodes}
        out = LI.test_body(session, learner, spread, args, np.random.default_rng(0), centres)
        assert set(session.calls) == kinds
        parts = {k: v for k, v in out.items() if not k.startswith("_")}
        assert len(parts) == 24 and all(p["horizon_steps"] == horizon for p in parts.values())
        if horizon > 1:
            assert out["_hold_execution_check"]["max_bone_difference"] == 0.0
            assert session.calls.count(("run_arm", 5)) == 1
        else:
            assert "_hold_execution_check" not in out
        assert LI.part_kinds({"0": out})["arm"]["tested"] == 2


# ------------------------------------------------------------------ the pause-job driver
def _driver():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "latent_full_body_driver", root / "docs" / "gain_of_function_v4" / "coordinator" / "latent_full_body_2026-10-01.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_driver_runs_the_hold_fit_as_the_primary_arm_only_when_its_line_passed(tmp_path):
    # Defects caught: the comparison run on a hold fit that failed its offline line; the one-step fit dropped although
    # it qualified; the tool's own records counted as parts.
    D = _driver()
    assert D.compare_plan(tmp_path)[0] is None
    (tmp_path / "fit").mkdir()
    (tmp_path / "verdict.json").write_text(json.dumps(dict(lines=dict(no_regression=dict(passed=True)))))
    (tmp_path / "fit" / "hierarchy.pt").write_bytes(b"x")
    plan, gates = D.compare_plan(tmp_path)
    assert plan is None and "witness bank" in gates["reason"]             # no goal source: skipped, not crashed
    (tmp_path / "hold_split").mkdir()
    (tmp_path / "hold_split" / "witness_bank.pt").write_bytes(b"x")
    (tmp_path / "fit" / "hierarchy.pt").write_bytes(b"x")
    (tmp_path / "verdict.json").write_text(json.dumps(dict(lines=dict(no_regression=dict(passed=True)))))
    plan, gates = D.compare_plan(tmp_path)
    assert plan["primary"][0] == tmp_path / "fit" / "hierarchy.pt" and "step" not in plan
    (tmp_path / "hold_fit_verdict.json").write_text(json.dumps(dict(lines=dict(hold_fit_future=dict(passed=True)))))
    assert D.compare_plan(tmp_path)[0]["primary"][0] == tmp_path / "fit" / "hierarchy.pt"   # no hold fit file yet
    (tmp_path / "hold_fit").mkdir()
    (tmp_path / "hold_fit" / "hierarchy.pt").write_bytes(b"x")
    plan, gates = D.compare_plan(tmp_path)
    assert plan["primary"][0] == tmp_path / "hold_fit" / "hierarchy.pt" and plan["step"][0] == tmp_path / "fit" / "hierarchy.pt"
    (tmp_path / "hold_fit_verdict.json").write_text(json.dumps(dict(lines=dict(hold_fit_future=dict(passed=False)))))
    assert D.compare_plan(tmp_path)[0]["primary"][0] == tmp_path / "fit" / "hierarchy.pt"
    counts = D.kind_counts(dict(bodies={"1": {"arm_L": dict(controllable=True), "_hold_execution_check": {}}}))
    assert counts == {"arm": dict(tested=1, controllable=1)}
    jobs = json.loads((Path(D.__file__).resolve().parents[1] / "pause_jobs_2026-10-01_latent_full_body.json").read_text(
        encoding="utf-8"))
    steps = [job["cmd"][job["cmd"].index("--step") + 1] for job in jobs]
    assert steps == ["fit", "hold_fit", "causal", "hold_causal", "compare"]
    assert all(job["need_gb"] > 0 and job["timeout_s"] > 0 for job in jobs)
