# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The lesson probe (LemLite.lesson_probe and LossManager.term_probe; review finding 3 of
docs/gain_of_function_v4/agents/proprio_token_review_2026-09-25.md): None in every run, so every recording site is a
no-op; with a probe installed, every term the real step adds to its loss is recorded, every lesson with its graph (a
plain 0 the step adds has none), and the recorded terms sum to the step's loss. docs/gain_of_function_v4/proprio_token_measurements.py grade uses it to grade
each lesson's pressure on the proprioception encoder."""
import inspect
import math


import pytest
import torch


def test_no_probe_in_a_run_and_every_loss_site_records():
    from mimo_bridge.lem_model import LemLite
    from lem.training.loss_manager import LossManager
    assert LemLite.lesson_probe is None and LossManager.term_probe is None
    step = inspect.getsource(LemLite.step)
    # every `loss = loss + <term>` of the step is followed, within its next five lines, by a recording of the term;
    # the next-moment lesson is recorded per sense where its parts are formed
    lines = step.splitlines()
    adds = [i for i, line in enumerate(lines) if "loss = loss + " in line]
    assert len(adds) >= 11, len(adds)
    for i in adds:
        if lines[i].strip() == "loss = loss + nf_loss":
            continue
        assert any("_probe_term(" in l for l in lines[i + 1:i + 6]), "not recorded: " + lines[i].strip()
    assert step.count('self._probe_term("next_frame." + _m') == 2, "both forms of the next-moment lesson"
    # the probe's two calls sit on either side of the backward and the clip
    assert step.index("self.lesson_probe.before_backward(self, loss)") < step.index("loss.backward()")
    assert step.index("self._clip_training_gradients()") < step.index("self.lesson_probe.after_clip(self, grad_norm)")
    compute = inspect.getsource(LossManager.compute)
    appends = compute.count("loss_terms.append(")
    assert appends == compute.count("self._probe("), "every summed term of the loss manager is recorded"


class Probe:
    def __init__(self):
        self.terms = {}
        self.seen = []
        self.norms = []

    def before_backward(self, model, loss):
        self.seen.append(dict(loss=float(loss.detach()),
                              terms={k: float(v.detach()) for k, v in self.terms.items()},
                              no_graph={k: float(v.detach()) for k, v in self.terms.items() if not v.requires_grad}))

    def after_clip(self, model, grad_norm):
        self.norms.append(float(grad_norm))


@pytest.mark.slow
def test_the_recorded_terms_sum_to_the_loss_of_the_real_step(tmp_path):
    torch.set_num_threads(1)
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.datasource import SingleEnvSource
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                overrides={"training": {"steps_per_phase": 3000}})
    probe = Probe()
    m.lesson_probe = probe
    m.loss_manager.term_probe = probe.terms
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "probe"), tag="_probe", vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(8):
            probe.terms.clear()
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()
    assert len(probe.seen) == 8 and len(probe.norms) == 8
    later = probe.seen[2:]                   # from the third step the delayed lessons are graded
    assert any(k.startswith("next_frame.") for s in later for k in s["terms"]), later[-1]["terms"]
    assert any(k.startswith("prediction.") or k == "vicreg" or k == "motor" for s in later for k in s["terms"])
    for i, s in enumerate(probe.seen):
        # A term without a graph is a plain number the step adds, never a detached copy of a lesson: VICReg's
        # temporal form returns a plain 0 on a sense's first frame, when no recent token exists to compare with
        # (lem/training/loss_manager.py compute_vicreg). Measured 2026-09-26: at the first step only, VICReg alone,
        # 0.0 (the first run of this test asked every term for a graph and failed there). The grade skips such a
        # term (proprio_token_measurements.GradeProbe): it has no gradient to grade.
        assert all(v == 0.0 for v in s["no_graph"].values()), (i, s["no_graph"])
        if i > 0:
            assert not s["no_graph"], "every term after the first step is recorded with its graph: %r" % s["no_graph"]
        total = sum(s["terms"].values())
        assert math.isclose(total, s["loss"], rel_tol=1e-5, abs_tol=1e-6), (total, s["loss"], s["terms"])
