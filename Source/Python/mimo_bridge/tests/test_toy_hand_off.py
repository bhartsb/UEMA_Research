# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The adult hands the toy over when the infant grips it (--toy-hand-off follow).

The toy is a MuJoCo mocap body. MuJoCo writes its place from outside every step and contact
can never move it - measured 2026-09-15, every infant actuator at full force for 1.0 s moved it
exactly 0 m - so nothing the infant did could carry it, and UEMA 11.2's grasped toy joining the
self could never be earned in the crib. These tests pin the hand-off, on the real crib scene.
"""
from pathlib import Path

import numpy as np
import pytest

SPEECH = Path(__file__).resolve().parents[4] / "Saved/mimo_bridge/caregiver_speech_exchange_v1_final"


def _env(tmp_path, hand_off):
    from mimo_bridge.env import BridgeEnv
    e = BridgeEnv(scene="crib", vision=False, audio_real=True, mobile=False, age=24,
                  caregiver_mode="responsive", caregiver_speech_dir=str(SPEECH),
                  run_dir=str(tmp_path), toy_hand_off=hand_off)
    e.reset()
    return e


def _into_the_hand(e):
    """Put the ball where at least two parts of the right hand touch it, and keep the adult's
    carry aiming there. Returns how many right-hand parts touch it."""
    c = e.caregiver
    m, d = e.model, e.data
    c.state["style"] = 0                       # the ball, whose shape collides
    c._appearance(e, 0)
    hand = c.hand_anchor["right"]
    base = np.asarray(d.xpos[hand]).copy()
    R = np.asarray(d.xmat[hand]).reshape(3, 3)
    best, best_n = None, -1
    for off in ([0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02], [0.03, 0.01, 0],
                [0.02, -0.01, 0.01], [0.04, 0, 0], [0.01, 0.02, 0.01]):
        p = base + R @ np.asarray(off, dtype=np.float64)
        d.mocap_pos[e._target_id] = p
        c.state["target_goal"] = p.copy()
        e._mj.mj_forward(m, d)
        n = len(c._hand_parts_on_toy(e)["right"])
        if n > best_n:
            best, best_n = p.copy(), n
    d.mocap_pos[e._target_id] = best
    c.state["target_goal"] = best.copy()
    e._mj.mj_forward(m, d)
    return best_n


def _grip(e, monkeypatch):
    """Two parts of the right hand on the toy, every step, until the adult hands it over.

    A mocap ball forced into a still hand pushes the hand out of itself within a step, so a
    placement alone does not make a lasting grip - closing the fingers around the toy is the
    policy's to learn. What these tests pin is what happens once a grip exists, so the contacts
    of a grip are supplied, from two real parts of the real right hand."""
    c = e.caregiver
    parts = {b for b, side in c.hand_parts.items() if side == "right"}
    two = set(sorted(parts)[:2])
    monkeypatch.setattr(type(c), "_hand_parts_on_toy",
                        lambda self, env: {"left": set(), "right": set(two)})
    for _ in range(3 * c.GRIP_STEPS):
        e.step(ctrl=np.zeros(e.model.nu))
        if (c.state.get("grip") or {}).get("held"):
            return True
    return False


def test_the_hand_parts_are_found_on_both_hands(tmp_path):
    e = _env(tmp_path, "follow")
    try:
        c = e.caregiver
        assert set(c.hand_anchor) == {"left", "right"}
        sides = list(c.hand_parts.values())
        assert sides.count("left") >= 5 and sides.count("right") >= 5, sides
    finally:
        e.close()


def test_with_the_hand_off_off_a_grip_never_moves_the_toy(tmp_path):
    e = _env(tmp_path, "off")
    try:
        assert _into_the_hand(e) >= 2
        rng = np.random.default_rng(0)
        for _ in range(60):
            e.step(ctrl=np.clip(0.4 * rng.standard_normal(e.model.nu), -1, 1))
        g = e.caregiver.state.get("grip") or {}
        assert g.get("grips", 0) == 0 and not g.get("held", False)
        assert e.caregiver.diagnostics(e)["object_follows_grip"] is False
    finally:
        e.close()


def test_two_parts_of_a_real_hand_can_touch_the_toy_at_once(tmp_path):
    """The contact rule reads the real scene: placed against the right hand, the ball is
    touched by at least two of that hand's parts."""
    e = _env(tmp_path, "follow")
    try:
        assert _into_the_hand(e) >= 2
    finally:
        e.close()


def test_a_grip_hands_the_toy_over_and_it_goes_where_the_hand_goes(tmp_path, monkeypatch):
    e = _env(tmp_path, "follow")
    try:
        c = e.caregiver
        assert _into_the_hand(e) >= 2
        assert _grip(e, monkeypatch), "the grip was not handed over"
        g = c.state["grip"]
        assert g["held"] and g["side"] == "right", g
        hand = c.hand_anchor["right"]
        d = e.data

        def local():
            R = np.asarray(d.xmat[hand]).reshape(3, 3)
            return R.T @ (np.asarray(d.mocap_pos[e._target_id]) - np.asarray(d.xpos[hand]))

        # EXACT WHERE THE ADULT PLACES IT: straight after the hand-off rule runs, before any
        # physics moves the hand, the toy sits at the grip's own pose in the hand's frame.
        c._grip_step(e)
        assert np.allclose(local(), np.asarray(g["local_pos"]), atol=1e-9), (
            "the placement rule does not put the toy at the grip's pose")
        start = np.asarray(d.mocap_pos[e._target_id]).copy()
        rng = np.random.default_rng(1)
        travelled, worst, biggest_step = 0.0, 0.0, 0.0
        prev = start.copy()
        for _ in range(80):
            e.step(ctrl=np.clip(0.4 * rng.standard_normal(e.model.nu), -1, 1))
            if not g["held"]:
                break
            now = np.asarray(d.mocap_pos[e._target_id]).copy()
            step = float(np.linalg.norm(now - prev))
            travelled += step
            biggest_step = max(biggest_step, step)
            prev = now
            worst = max(worst, float(np.linalg.norm(local() - np.asarray(g["local_pos"]))))
        # AND WITHIN ONE STEP OF IT AFTER THE PHYSICS. The adult places the toy partway through
        # a step and the hand goes on moving and turning for the rest of it, so the toy trails
        # by at most what its own place in the hand travels in one step - which includes the
        # hand's turning times the 4.7 cm the toy sits from the hand's origin. The pose is
        # recomputed from the grip every step, so the trail never accumulates into a slip.
        assert worst <= 2.0 * biggest_step + 1e-4, (
            "the toy trailed the hand by %.4f m while its largest one-step travel was %.4f m"
            % (worst, biggest_step))
        assert travelled > 0.005, (
            "the toy was held but did not go anywhere: %.4f m" % travelled)
        diag = c.diagnostics(e)
        assert diag["object_follows_grip"] is True and diag["object_grips"] >= 1
    finally:
        e.close()


def test_letting_go_gives_the_toy_back_to_the_adult(tmp_path, monkeypatch):
    e = _env(tmp_path, "follow")
    try:
        c = e.caregiver
        assert _into_the_hand(e) >= 2
        assert _grip(e, monkeypatch)
        assert c.state["grip"]["held"]
        # the hand opens: no part of it touches the toy any more
        monkeypatch.setattr(type(c), "_hand_parts_on_toy",
                            lambda self, env: {"left": set(), "right": set()})
        for _ in range(c.RELEASE_STEPS - 1):
            e.step(ctrl=np.zeros(e.model.nu))
        assert c.state["grip"]["held"], "released before the hand had been off it long enough"
        e.step(ctrl=np.zeros(e.model.nu))
        g = c.state["grip"]
        assert not g["held"] and g["releases"] == 1
        assert g["blend"] > 0, "the adult's hand should be on its way back to the toy"
    finally:
        e.close()


def test_the_adults_hand_stays_where_it_let_go_while_the_infant_holds_the_toy(tmp_path,
                                                                               monkeypatch):
    e = _env(tmp_path, "follow")
    try:
        c = e.caregiver
        assert _into_the_hand(e) >= 2
        assert _grip(e, monkeypatch)
        assert c.state["grip"]["held"]
        let_go = np.asarray(c.state["grip"]["adult_hand"])
        rng = np.random.default_rng(2)
        for _ in range(20):
            e.step(ctrl=np.clip(0.4 * rng.standard_normal(e.model.nu), -1, 1))
            if not c.state["grip"]["held"]:
                break
            assert np.allclose(e.data.mocap_pos[c.ids["caregiver_hand_l"]], let_go, atol=1e-9), (
                "the adult's hand rode along with a toy the infant was holding")
    finally:
        e.close()


def test_the_report_says_the_body_can_move_the_toy_once_it_is_handed_over():
    from mimo_bridge.lem_model import object_physics_driven_for
    assert object_physics_driven_for({"caregiver": {"object_physics_driven": False,
                                                    "object_follows_grip": True}})["toy"] is True
    assert object_physics_driven_for({"caregiver": {"object_physics_driven": False,
                                                    "object_follows_grip": False}})["toy"] is False
    assert object_physics_driven_for({})["toy"] is False


def test_an_unknown_hand_off_setting_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _env(tmp_path, "sometimes")


def test_the_trainer_flag_reaches_the_world():
    from mimo_bridge import train_mimo as TM
    args = TM.parse_args(["--toy-hand-off", "follow", "--scene", "crib", "--out", "x"])
    assert args.toy_hand_off == "follow"
    assert TM.env_kwargs_for(args, None, "x")["toy_hand_off"] == "follow"
    assert TM.parse_args(["--scene", "crib", "--out", "x"]).toy_hand_off == "off"
