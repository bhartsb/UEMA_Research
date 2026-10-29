# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The whole-body hierarchy through its tools (Oct 1, 2026): the split's part scales, the causal-control test of every
part, the comparison's goals and its owner-predicted, per-handle hold score. CPU, no world (stand-in sessions)."""
import json
import types

import numpy as np
import pytest
import torch

from lem.latent_body.learning import ShadowLearner
from lem.latent_body.trajectory import PART_GROUPS, eye_quaternion
from lem.latent_body.tree import arm_hand_tree, full_body_tree
from mimo_bridge.body_layout import names_for
from mimo_bridge.tests.test_latent_full_body_rows import CONTRACT, NAMES, _example, config
from mimo_bridge.tools import latent_body_compare as LC
from mimo_bridge.tools import latent_body_intervention as LI


def _row(ex, body):
    return dict(history=dict(ex.history.state_dict(), env_id=body), start=dict(ex.start.state_dict(), env_id=body),
                end=dict(ex.end.state_dict(), env_id=body), executed=ex.executed, action_contract=ex.action_contract)


def test_the_split_measures_each_part_groups_own_scales(tmp_path):
    # Defect caught: one pooled scale for every group (the arm's 0.31 cm step against the hand's 0.92 cm in the
    # recorded rows); unrecorded contact divided by the floor; the eyes' position change scaled by float noise.
    from mimo_bridge.tools import latent_body_split as S
    path = tmp_path / "t.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for k in range(60):
            ex = _example(k)
            g = torch.Generator().manual_seed(k)
            ex.end.bones = ex.start.bones.clone()
            ex.end.bones[:, 4:7] += torch.randn(26, 3, generator=g) * 0.1               # everything moves a little
            leg = [NAMES.index(n) for n in ("thigh_L", "calf_L", "foot_L", "thigh_R", "calf_R", "foot_R")]
            ex.end.bones[leg, 4:7] += 5.0 * (k % 3)                                      # the legs move far
            ex.end.sim_time, ex.end.frame = ex.start.sim_time + .05, ex.start.frame + 1
            f.write(json.dumps(_row(ex, k % 2), default=lambda t: t.tolist()) + "\n")
    base = config()
    base.pop("body_rows"), base.pop("digit_frame"), base.pop("hand_action_frame")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    S.main(["--trajectory", str(path), "--holdout-body", "1", "--base-config", str(tmp_path / "base.json"),
            "--output", str(tmp_path / "out"), "--levels", "full_body", "--eye-row-convention", "legacy",
            "--contact-recorded-levels", "arm,hand,index", "--scale-rows", "30"])
    cfg = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    scales = cfg["body_rows"]["part_scales"]
    assert set(scales) == set(PART_GROUPS)
    assert scales["leg"]["change"][0] > 2 * scales["digit"]["change"][0]
    assert scales["gaze"]["change"][:3] == [1.0, 1.0, 1.0]
    assert scales["leg"]["state"][11] == 1.0                                 # leg contact never recorded
    assert sorted(cfg["body_rows"]["contact_recorded"]) == sorted(arm_hand_tree(NAMES).owners)
    assert cfg["digit_frame"] == "wrist" and cfg["body_rows"]["future_rows"] == "own"
    ShadowLearner(full_body_tree(NAMES), cfg)


class _Session:
    """A stand-in world: each handle moves half way to its ordered position in one step; the eyes report the gaze
    they were sent."""
    def __init__(self):
        from types import SimpleNamespace
        from lem.model.contact_head import PATCH_NAMES
        self.bones = _example(1).start.bones.numpy().copy()
        self.patches = [0] * len(PATCH_NAMES)
        self.gazes = []
        self.model = SimpleNamespace(handle_names=NAMES, _gaze_from_handle=self._gaze)
        self.t = 0.
        self.frames = [self._frame()]

    @staticmethod
    def _gaze(order):
        from lem.latent_body.trajectory import eye_yaw_pitch
        yaw, pitch = eye_yaw_pitch(torch.as_tensor(order)[:, NAMES.index("gaze"), 3:7])
        return torch.stack([yaw, pitch], -1).numpy()

    def _frame(self):
        from types import SimpleNamespace
        return SimpleNamespace(env_index=0, sim_time=self.t, bones=self.bones.copy(),
                               meta={"task": {"contact_patch_state": list(self.patches)},
                                     "hand_target_contract": "wrist_relative_joint_v1"})

    def policy_act(self):
        pass

    def observe(self):
        self.t += .05
        self.frames = [self._frame()]

    def snapshot(self):
        return (self.bones.copy(), self.t)

    def hold_order(self):
        order = np.zeros((1, len(self.bones), 9), dtype=np.float32)
        order[0, :, :3], order[0, :, 3:7], order[0, :, 8] = self.bones[:, 4:7], self.bones[:, :4], .5
        return order

    def run_arm(self, snap, order, steps, gaze_rad=None):
        bones, t = snap
        self.gazes.append(gaze_rad)
        self.bones = bones.copy()
        self.bones[:, 4:7] += .5 * (order[0, :, :3] - bones[:, 4:7])
        g = NAMES.index("gaze")
        if gaze_rad is not None:      # the eye turns to what it was sent, in the physical convention
            self.bones[g, :4] = eye_quaternion(torch.tensor(float(gaze_rad[0])), torch.tensor(-float(gaze_rad[1])))
        self.t = t + .05 * steps
        self.frames = [self._frame()]


def test_every_part_of_the_whole_body_is_tested_and_the_eyes_receive_their_command(monkeypatch):
    # Defects caught: a part left out of the causal test; the eyes held while their code changes (the gaze part could
    # then never pass); the head, eyes and feet graded on a displacement their hinges barely make.
    import mimo_bridge.latent_body_runtime as RT
    monkeypatch.setattr(RT, "observed_contract", lambda frame: CONTRACT)
    learner = ShadowLearner(full_body_tree(NAMES), config())
    session = _Session()
    centres = {n.name: torch.zeros(1, learner.model.latent_dim) for n in learner.tree.nodes}
    spread = {n.name: torch.ones(learner.model.latent_dim) for n in learner.tree.nodes}
    args = types.SimpleNamespace(warmup_steps=2, codes=4, centre="mean", seed=0)
    out = LI.test_body(session, learner, spread, args, np.random.default_rng(0), centres)
    assert set(out) == {n.name for n in learner.tree.nodes} and len(out) == 24
    assert all(g is not None and len(g) == 2 for g in session.gazes)
    graded = {name: part["graded_on"] for name, part in out.items()}
    assert {n for n, g in graded.items() if g == "turn"} == {"head", "gaze", "foot_L", "foot_R"}
    assert all(part["repeat_cm"] == 0 and part["repeat_rad"] == 0 for part in out.values())
    kinds = LI.part_kinds({"0": out})
    assert kinds["index"] == dict(tested=2, controllable=sum(out[f"index_{s}"]["controllable"] for s in "LR"))
    # the pilot still holds the eyes (run_arm without a gaze) and grades every part on displacement
    pilot = ShadowLearner(arm_hand_tree(NAMES), _pilot_config())
    session = _Session()
    out = LI.test_body(session, pilot, {n.name: torch.ones(2) for n in pilot.tree.nodes}, args,
                       np.random.default_rng(0), {n.name: torch.zeros(1, 2) for n in pilot.tree.nodes})
    assert all(g is None for g in session.gazes) and {p["graded_on"] for p in out.values()} == {"move"}


def _pilot_config():
    cfg = config()
    cfg.pop("body_rows")
    return cfg


def test_the_comparison_adds_trunk_and_leg_goals_for_the_whole_body():
    # the Sep 30 goals (anchored_v1); the redesigned goals: test_latent_compare_design.py
    a = "anchored_v1"
    assert set(LC.declared_goals(arm_hand_tree(NAMES), design=a)) == {"wrist_L_reach", "index_L_curl", "index_R_curl"}
    assert set(LC.declared_goals(full_body_tree(NAMES), design=a)) == set(LC.GOALS) and len(LC.GOALS) == 5
    with pytest.raises(ValueError):
        LC.declared_goals(arm_hand_tree(NAMES), "knee_L_flex", design=a)
    assert LC.wins_needed(12) == 10 and LC.wins_needed(20) == 15 and LC.wins_needed(18) == 13


def test_the_whole_body_score_uses_each_handles_owner_and_hold_factor(monkeypatch):
    # Defects caught: the root's ungraded descendant rows scored; one hold factor for every handle; the eyes' row
    # placed (the comparison keeps them matched; latent_order itself returns every owned row).
    import lem.planner.latent_candidates as LCand
    from mimo_bridge.tests.test_latent_full_body_rows import _obs
    learner = ShadowLearner(full_body_tree(NAMES), config())
    ex = _example(2)
    ex.history, ex.start = _obs(0, 4), _obs(1, 5)
    tree = learner.tree
    target = ex.start.bones[:, 4:7].numpy() + 1.0
    template = torch.zeros(26, 9)
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    seen = {}
    real = LCand.search

    def spy(model, state, history, score, **kw):
        seen["score"] = score
        return real(model, state, history, score, **kw)
    monkeypatch.setattr(LCand, "search", spy)
    order, _, receipt = LC.latent_order(learner, ex.history, ex.start, target, template, candidates=4, radius=.5,
                                        generator=torch.Generator().manual_seed(0))
    assert receipt["hold_factor"] == "HOLD_FACTORS"
    # every part predicts 1 cm per axis / sqrt(3) on its own rows (relative rows, the digits, the eyes and the carried
    # rows: no movement relative to their parent, so they are carried the same 1 cm); ungraded descendant rows predict
    # nothing
    from lem.latent_body.trajectory import GROUP_OF, row_options
    body = row_options(learner.config)["body"]
    relative = {i for i, n in enumerate(NAMES) if GROUP_OF[n] == "digit" or n in body.positionless}
    assert body.positionless == {"gaze", "chest", "upperarm_L", "upperarm_R", "calf_L", "calf_R", "foot_L", "foot_R"}
    scales = learner.config["body_rows"]["part_scales"]
    preds = {}
    for n in tree.nodes:
        rows = torch.zeros(len(tree.subtree[n.name]), 6)
        for k, i in enumerate(tree.subtree[n.name]):
            if i in tree.owned[n.name] and i not in relative:
                rows[k, :3] = (1 / np.sqrt(3)) / torch.tensor(scales[GROUP_OF[NAMES[i]]]["change"][:3])
        preds[n.name] = types.SimpleNamespace(change=rows.flatten().repeat(4, 1))
    got = seen["score"](preds)
    factor = lambda name: LC.HOLD_FACTORS[name[:-2] if name[-2:] in ("_L", "_R") else name]
    expected = np.mean([3 * (factor(NAMES[i]) / np.sqrt(3) - 1) ** 2 for r in tree.roots for i in tree.subtree[r]])
    assert torch.allclose(got, torch.full((4,), float(expected)), atol=1e-3)
