# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The touch layout's repeatability switch (mimo_bridge/touch_inside_rays.py; the defect and its measurement:
docs/gain_of_function_v4/agents/training_repeatability_2026-09-26.md).

MIMo's touch drops the skin points that lie inside a neighbouring shape of the same body part, testing each point
with rays in 10 directions drawn once per process from an unseeded generator. 'process' (default) keeps that;
'fixed_v1' builds every world's touch with one fixed set, so the live skin points no longer depend on the process.
The world test forces MIMo's process draw to generator seed 571, one of the 20 in 20,000 measured draws that give
the skin proof's divergent layout (8 extra live points, 4 on each hand), and to seed 0, which gives the usual layout.
"""
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge import continuation_state as CS
from mimo_bridge import touch_inside_rays as TIR

TRAIN = Path(__file__).resolve().parents[1] / "train_mimo.py"


def _mimo_draw(k):
    import mimo_infant.simulation.utils as mu
    return mu.normalize_vectors(np.random.default_rng(k).normal(size=(10, 3)))


def _points_on_a_box_face():
    """A unit box and points on its top face, inside it and outside it: the top-face points are the seam case."""
    import trimesh
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    pts = np.array([[0.1, 0.2, 0.5], [-0.3, 0.1, 0.5], [0.0, 0.0, 0.0], [0.2, -0.2, 0.3], [0.0, 0.0, 0.9]])
    return box, pts


def test_fixed_v1_is_mimos_own_recipe_with_seed_0_written_out():
    assert TIR.FIXED_V1.shape == (10, 3) and TIR.FIXED_V1.dtype == np.float64 and not TIR.FIXED_V1.flags.writeable
    assert np.allclose(np.linalg.norm(TIR.FIXED_V1, axis=1), 1.0, rtol=0, atol=1e-15)
    assert hashlib.sha256(np.ascontiguousarray(TIR.FIXED_V1).tobytes()).hexdigest() == TIR.FIXED_V1_SHA256
    # the recipe itself on this numpy (touch.py:1937-1938 with seed 0); the literal stays if a numpy release moves it
    assert np.array_equal(_mimo_draw(0), TIR.FIXED_V1)


def test_the_block_runs_mimos_own_test_with_the_fixed_directions_whatever_a_caller_passes_and_restores_it():
    import mimo_infant.touch.touch as mt
    own, own_defaults = mt._contains, mt._contains.__defaults__
    box, pts = _points_on_a_box_face()
    expected = own(pts, box, TIR.FIXED_V1.copy(), own_defaults[1])
    # seed 571's 10 directions all point down, so without the switch the top-face points read outside: a draw that
    # would change the answer, which the switch must ignore
    assert not np.array_equal(own(pts, box, _mimo_draw(571), own_defaults[1]), expected)
    with TIR.inside_rays("fixed_v1") as use:
        assert mt._contains is not own
        assert np.array_equal(mt._contains(pts, box), expected)
        # directions a caller passes do not reach the test: the switch's directions are the fixed ones
        assert np.array_equal(mt._contains(pts, box, _mimo_draw(571)), expected)
        assert np.array_equal(mt._contains(pts, box, directions=_mimo_draw(571)), expected)
        assert use.calls == 3
    assert mt._contains is own and mt._contains.__defaults__ is own_defaults
    assert use.report() == dict(setting="fixed_v1", inside_tests=3, directions_sha256=TIR.FIXED_V1_SHA256)
    with pytest.raises(KeyError):
        with TIR.inside_rays("fixed_v1"):
            raise KeyError("a build that fails")
    assert mt._contains is own
    with TIR.inside_rays("process") as use:
        assert mt._contains is own
    assert mt._contains is own and use.report() == dict(setting="process", inside_tests=0, directions_sha256=None)


def test_a_seam_point_is_outside_only_when_every_ray_points_into_the_box():
    """The defect's rule on a plain box: a point on the top face counts as inside unless all 10 rays point down."""
    import mimo_infant.touch.touch as mt
    box, pts = _points_on_a_box_face()
    rng = np.random.default_rng(3)
    down = rng.normal(size=(10, 3))
    down[:, 2] = -np.abs(down[:, 2]) - 0.1
    down /= np.linalg.norm(down, axis=1, keepdims=True)
    one_up = down.copy()
    one_up[4, 2] = abs(one_up[4, 2])
    inside_down = mt._contains(pts, box, down)
    inside_one_up = mt._contains(pts, box, one_up)
    assert list(inside_down[:2]) == [False, False] and list(inside_one_up[:2]) == [True, True]
    assert inside_down[2] and inside_one_up[2] and inside_down[3] and not inside_down[4]


def test_an_unknown_setting_is_refused_by_the_block_the_world_and_a_resume(tmp_path):
    with pytest.raises(ValueError, match="touch_inside_rays"):
        with TIR.inside_rays("seeded"):
            pass
    from mimo_bridge.env import BridgeEnv
    with pytest.raises(ValueError, match="touch inside rays"):
        BridgeEnv(scene="crib", vision=False, run_dir=str(tmp_path), touch_inside_rays="seeded")
    state = dict(version=CS.VERSION, worlds=[dict(env_kwargs=dict(scene="crib"), world={}, payload={})],
                 config={}, trainer=dict(step=1))
    with pytest.raises(ValueError, match="touch_inside_rays"):
        CS.migrate_env_kwargs(state, SimpleNamespace(touch_inside_rays="seeded"), env_settings="launch")


def test_a_changed_mimo_inside_test_is_refused_not_silently_ignored(monkeypatch):
    import mimo_infant.touch.touch as mt

    def changed(points, mesh, rays=None):
        return None

    monkeypatch.setattr(mt, "_contains", changed)
    with pytest.raises(RuntimeError, match="changed shape"):
        with TIR.inside_rays("fixed_v1"):
            pass


def test_a_build_the_switch_did_not_reach_for_every_shape_pair_is_refused():
    import trimesh
    two = SimpleNamespace(_submeshes={5: [trimesh.creation.box(), trimesh.creation.icosphere()]})
    none = SimpleNamespace(_submeshes={5: [trimesh.creation.box()]})
    assert TIR.shapes_to_test(two) == 2 and TIR.shapes_to_test(none) == 0
    for calls in (0, 1):                                # none or some pairs bypassed the switch
        use = TIR.InsideRayUse("fixed_v1")
        use.calls = calls
        with pytest.raises(RuntimeError, match="ran %d inside tests while MIMo's touch has 2 shape pairs" % calls):
            TIR.check_reached(use, two)
    TIR.check_reached(TIR.InsideRayUse("fixed_v1"), none)      # nothing to test: nothing to refuse
    TIR.check_reached(TIR.InsideRayUse("process"), two)
    used = TIR.InsideRayUse("fixed_v1")
    used.calls = 2
    TIR.check_reached(used, two)


def test_the_trainer_flag_defaults_to_process_and_names_the_world_setting_only_when_changed():
    src = TRAIN.read_text(encoding="utf-8")
    flag = re.search(r"--touch-inside-rays['\"].{0,120}?default=['\"](\w+)['\"]", src, re.S)
    assert flag and flag.group(1) == "process"
    from mimo_bridge import train_mimo as TM
    args = TM.parse_args(["--out", "unused"])
    assert args.touch_inside_rays == "process" and "touch_inside_rays" not in TM.env_kwargs_for(args, None, "out")
    args = TM.parse_args(["--out", "unused", "--touch-inside-rays", "fixed_v1"])
    assert TM.env_kwargs_for(args, None, "out")["touch_inside_rays"] == "fixed_v1"


def test_a_resume_takes_the_switch_only_when_asked_and_never_writes_the_default():
    worlds = [dict(env_kwargs=dict(scene="crib", seed=7 + i, tag="_w%d" % i), world={}, payload=dict(sim_time=1.0))
              for i in range(2)]
    state = dict(version=CS.VERSION, worlds=worlds, config={}, trainer=dict(step=10))
    for mode in ("launch", "saved"):
        up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(touch_inside_rays="process"), env_settings=mode)
        assert up is state and changes == []
    kept, none = CS.migrate_env_kwargs(state, SimpleNamespace(touch_inside_rays="fixed_v1"))
    assert kept is state and none == []
    up, changes = CS.migrate_env_kwargs(state, SimpleNamespace(touch_inside_rays="fixed_v1"), env_settings="launch")
    assert [c["new"] for c in changes] == ["fixed_v1"] * 2
    assert all(w["env_kwargs"]["touch_inside_rays"] == "fixed_v1" for w in up["worlds"])
    assert "touch_inside_rays" not in state["worlds"][0]["env_kwargs"]


def test_the_switch_makes_a_worlds_skin_layout_independent_of_the_process_draw(tmp_path, monkeypatch):
    """H84's body (crib, age 24, trunk_v1, mechanical_v4). MIMo's own directions are set to seed 571's draw, then
    to seed 0's; under 'process' the two builds differ by the 8 seam points, under 'fixed_v1' both give seed 0's
    layout: 4,683 skin points, digest 0581069917fae467, the layout 11 of the skin proof's 12 worlds had. The world
    reports every inside test MIMo made (58 shape pairs on this body) as run with the fixed directions."""
    import mimo_infant.touch.touch as mt
    from mimo_bridge.env import BridgeEnv

    def build(draw, setting):
        monkeypatch.setattr(mt._contains, "__defaults__", (_mimo_draw(draw), 1e-10))
        env = BridgeEnv(scene="crib", vision=False, run_dir=str(tmp_path / ("%d_%s" % (draw, setting))), age=24.0,
                        body_layout="trunk_v1", contact_profile="mechanical_v4", touch_inside_rays=setting)
        try:
            skin = [s.body_id for s in env.pooler.specs]
            names = {b: env.model.body(b).name for b in skin}
            out = (TIR.skin_layout(env.touch, skin), {names[b]: env.touch.get_sensor_count(b) for b in skin},
                   dict(env.touch_inside_rays_report), TIR.shapes_to_test(env.touch))
            assert np.array_equal(mt._contains.__defaults__[0], _mimo_draw(draw))   # MIMo's own draw is back
            return out
        finally:
            env.close()

    usual, usual_parts, usual_report, _ = build(0, "process")
    outlier, outlier_parts, _, _ = build(571, "process")
    fixed, fixed_parts, fixed_report, pairs = build(571, "fixed_v1")
    assert usual == (4683, "0581069917fae467") and outlier == (4691, "2daece73bcc3994c")
    assert {p: outlier_parts[p] - usual_parts[p] for p in usual_parts if outlier_parts[p] != usual_parts[p]} == {
        "right_hand": 2, "right_lfmetacarpal": 2, "left_hand": 2, "left_lfmetacarpal": 2}
    assert fixed == usual and fixed_parts == usual_parts
    assert usual_report == dict(setting="process", inside_tests=0, directions_sha256=None, skin_points=4683,
                                skin_digest="0581069917fae467")
    assert pairs == 58 and fixed_report == dict(setting="fixed_v1", inside_tests=58,
                                                directions_sha256=TIR.FIXED_V1_SHA256, skin_points=4683,
                                                skin_digest="0581069917fae467")
