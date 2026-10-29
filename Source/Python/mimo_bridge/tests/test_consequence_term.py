# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The consequence-access term in the goal-ring value (2026-09-18, tree revision W2/W1).

Until this term the ring held only lived moments ranked by five formulas that read no
contact, no moved object and no gained view, so a first touch of the mobile could not
become the goal. consequence_kinds counts, from the world's own task flags, how many
kinds of self-relevant consequence a moment carried (0-3); the launch weight
multiplies it into the stored value. Off (weight 0) is bit-identical to before.
"""
import torch

from mimo_bridge.lem_model import LemLite, consequence_kinds


def test_consequence_kinds_counts_the_three_kinds_from_the_world_flags():
    assert consequence_kinds({}) == 0.0
    assert consequence_kinds(None) == 0.0
    assert consequence_kinds({"mobile_touched_hand": True, "mobile_touched": True}) == 2.0   # contact + moved
    assert consequence_kinds({"mobile_touched": True}) == 1.0                                # a kick moves it
    assert consequence_kinds({"hand_pressing": True}) == 1.0
    assert consequence_kinds({"gaze_on_ball": True}) == 1.0
    assert consequence_kinds({"hand_in_fovea": True, "ball_touched_hand": True,
                              "ball_touched": True}) == 3.0
    # a flag that is present but False counts nothing
    assert consequence_kinds({"mobile_touched_hand": False, "gaze_on_ball": 0}) == 0.0


class _Frame:
    def __init__(self, meta):
        self.meta = meta


def _ring_value_with(model, frames_meta, weight):
    """Drive only the arithmetic the ring block performs on the consequence term."""
    model.consequence_weight = weight
    frames = [_Frame({"task": t}) for t in frames_meta]
    cq = torch.tensor([consequence_kinds(f.meta.get("task") or {}) for f in frames], dtype=torch.float32)
    return model.consequence_weight * cq


def test_the_term_is_zero_when_off_and_weight_per_kind_when_on():
    model = LemLite(device="cpu", seed=0, imagination=False, intention=True, intention_execute=False,
                    consequence_weight=0.0)
    metas = [{"mobile_touched_hand": True, "mobile_touched": True}, {}]
    assert torch.equal(_ring_value_with(model, metas, 0.0), torch.zeros(2))
    v = _ring_value_with(model, metas, 0.5)
    assert v.tolist() == [1.0, 0.0]        # two kinds x 0.5 for body 0, nothing for body 1


def test_a_step_with_a_consequence_records_it_and_raises_that_bodys_ring_value(tmp_path):
    """Behavioural: run the model on a real single-body world, plant the world's flags on
    one step, and read last_consequence and the ring: the planted body's stored value
    must exceed what the same step would have stored without the term by exactly
    weight x kinds."""
    from mimo_bridge.datasource import SingleEnvSource
    model = LemLite(device="cpu", seed=0, imagination=False, intention=True, intention_execute=False,
                    next_frame_loss=True, consequence_weight=0.5)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "cq"), tag="_cq", vision=False, seed=7)
    try:
        frames = src.reset()
        dev = dict(model.dev_limits(10 ** 9))
        held = prev = None
        for i in range(4):
            if i == 3:
                for f in frames:
                    f.meta.setdefault("task", {}).update(mobile_touched_hand=True, mobile_touched=True,
                                                         gaze_on_ball=True)
            # what the world itself reported this step. Not assumed zero: since the fingers
            # can move (2026-09-22) the hands press on the body within two steps, which the
            # world flags as hand_pressing - one kind - with nothing planted.
            reported = consequence_kinds(frames[0].meta.get("task") or {})
            res = model.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i == 3:
                assert model.last_consequence["kinds_mean"] == 3.0
                assert model.last_consequence["rate"] == 1.0
                # stored value = (w0 + drives + 0.5*3) x (1 + drive); the bodily value w0
                # and the drives are the same step's numbers, so the added term is what
                # separates the stored value from the value before the term
                stored = float(model.goal_ring[-1][1][0])
                w0 = float(model._goal_ring_w0[-1][0])
                assert stored - w0 >= 1.5 - 1e-5, (stored, w0)
                assert float(model._goal_ring_cq[-1][0]) == 3.0
            else:
                assert float(model._goal_ring_cq[-1][0]) == reported
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()


def test_the_flag_is_part_of_the_acting_configuration_not_operational():
    import inspect
    from mimo_bridge import train_mimo as TM
    src = inspect.getsource(TM)
    op_start = src.index("operational={")
    op_block = src[op_start:src.index("}", op_start)]
    assert "consequence_w" not in op_block
    assert "refresh_action_calibration" in op_block      # a resume-time action, never a fingerprint
