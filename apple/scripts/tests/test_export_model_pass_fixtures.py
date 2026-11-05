# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The model-pass fixtures (apple/scripts/export_model_pass_fixtures.py, WI-34) hold what the Swift model port needs.

Without the model or a world: the call recorder (a forward hook with kwargs and a method shadow that records and
restores), hook_model on a stand-in model (which methods are shadowed, which modules hooked, what remove()
restores), the acceptance check (six 192-number tokens, the five predicted tokens of the receipt's prediction heads,
no NaN, no predicted token for a sense without a head), the dropout reader, the encoder-input reader, the
safetensors round trip with metadata. With the fixtures on disk (skipped with the command that writes them when
they are absent): three states; the six tokens and five predicted tokens at 192 numbers; the command [1, 26, 9];
the wellness sum equal to the sum of its channels within 1e-5; the context zeros and the fed tokens equal to the
raw tokens (the first model step after a reset); the persistent memory's bias present; the metadata's senses,
prediction senses and the checkpoint's model.safetensors sha256 (checked against the file when it is here). Run
from Source/Python:
    python -m pytest ../../apple/scripts/tests/test_export_model_pass_fixtures.py -p no:cacheprovider -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import export_model_pass_fixtures as emp  # noqa: E402

SENSES = ("vision", "tactile", "audio", "proprioception", "motor", "vestibular")
HEADS = ("vision", "tactile", "audio", "proprioception", "vestibular")


def _good_tensors():
    t = {"token/%s" % s: np.ones(192, dtype=np.float32) for s in SENSES}
    t.update({"predicted/%s" % s: np.zeros(192, dtype=np.float32) for s in HEADS})
    t["command/motor_out"] = np.zeros((1, 26, 9), dtype=np.float32)
    return t


def test_check_fixture_passes_six_tokens_and_five_predicted_tokens():
    assert emp.check_fixture(_good_tensors()) == []
    assert emp.PREDICTION_SENSES == HEADS and emp.SENSES == SENSES and emp.D_MODEL == 192


def test_check_fixture_names_every_failure():
    t = _good_tensors()
    del t["token/motor"]
    t["predicted/vision"] = np.zeros(191, dtype=np.float32)
    t["predicted/motor"] = np.zeros(192, dtype=np.float32)
    t["command/motor_out"][0, 3, 4] = np.nan
    bad = emp.check_fixture(t)
    assert any("token/motor missing" in b for b in bad)
    assert any("predicted/vision has shape (191,)" in b for b in bad)
    assert any("without a prediction head" in b and "predicted/motor" in b for b in bad)
    assert any("command/motor_out holds a non-finite number" in b for b in bad)
    assert len(bad) == 4
    t2 = _good_tensors()
    t2["token/audio"][5] = np.inf
    assert emp.check_fixture(t2) == ["token/audio holds a non-finite number"]


def test_call_recorder_hooks_and_method_shadows_record_then_restore():
    torch = pytest.importorskip("torch")

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)

        def backbone_forward(self, x):
            return self.lin(x)

        def head_forward(self, f):
            return f.sum(-1, keepdim=True)

        def forward(self, x, scale=1.0):
            return self.head_forward(self.backbone_forward(x)) * scale

    enc = Enc()
    rec = emp.CallRecorder()
    rec.attach("enc", enc)
    for attr in ("forward", "backbone_forward", "head_forward"):
        rec.attach_method("enc." + attr, enc, attr)
    x = torch.ones(1, 3)
    out = enc(x, scale=2.0)
    assert rec.counts() == {"enc": 1, "enc.forward": 1, "enc.backbone_forward": 1, "enc.head_forward": 1}
    args, kwargs, hooked_out = rec.of("enc")[0]
    assert args[0] is x and kwargs == {"scale": 2.0} and hooked_out is out
    assert rec.of("enc.forward")[0][1] == {"scale": 2.0}
    assert torch.equal(rec.of("enc.backbone_forward")[0][2], enc.lin(x))
    rec.remove()
    assert "forward" not in enc.__dict__ and "backbone_forward" not in enc.__dict__
    before = len(rec.calls)
    enc(x)
    assert len(rec.calls) == before                # nothing records after remove()
    assert torch.equal(enc(x), Enc.forward(enc, x))


def test_hook_model_shadows_encoder_and_side_methods_and_hooks_modules():
    """hook_model on a stand-in model: the encoders' forward, backbone_forward and head_forward are shadowed, the
    output LayerNorm, forward model, heads, controller, wellness and the torch-module side modules get forward
    hooks, the side methods persistent_memory.encode, .bias and self_state.perceive are shadowed; remove()
    restores every method and stops recording."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)

        def backbone_forward(self, x):
            return self.lin(x)

        def head_forward(self, f):
            return f.sum(-1, keepdim=True)

        def forward(self, x):
            return self.head_forward(self.backbone_forward(x))

    class Plain(torch.nn.Module):          # an encoder with forward only
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.lin(x)

    class Memory:
        def encode(self, events, observe=False):
            return ("state", observe)

        def bias(self, state, site=None):
            return (state, site)

    class SelfState:
        def perceive(self, **inputs):
            return {"bias": len(inputs)}

    encoders = {"motor": Enc(), "proprioception": Plain()}
    model = SimpleNamespace(
        encoder_registry=SimpleNamespace(encoders=encoders, output_norm=torch.nn.LayerNorm(1)),
        forward_model=torch.nn.Identity(), prediction_heads=torch.nn.Identity(), motor_controller=torch.nn.Identity(),
        wellness=torch.nn.Identity(), persistent_memory=Memory(), self_state=SelfState(),
        heads_all_slot_path=torch.nn.Identity(), motor_asked_change=None, liveness=None)
    rec, modules = emp.hook_model(model)
    assert set(modules) == {"encoder_registry.output_norm", "forward_model", "prediction_heads", "motor_controller",
                            "wellness", "heads_all_slot_path"}
    x = torch.ones(1, 3)
    model.encoder_registry.output_norm(encoders["motor"](x))
    model.encoder_registry.output_norm(encoders["proprioception"](x))   # no backbone_forward: only forward is shadowed
    model.forward_model(x)
    state, _ = model.persistent_memory.encode([1, 2], observe=True)
    model.persistent_memory.bias(state, site="lived")
    model.self_state.perceive(a=1, b=2)
    assert rec.counts() == {"encoder.motor.forward": 1, "encoder.motor.backbone_forward": 1,
                            "encoder.motor.head_forward": 1, "encoder.proprioception.forward": 1,
                            "encoder_registry.output_norm": 2, "forward_model": 1,
                            "persistent_memory.encode": 1, "persistent_memory.bias": 1, "self_state.perceive": 1}
    assert rec.of("persistent_memory.encode")[0][1] == {"observe": True}
    assert rec.of("persistent_memory.bias")[0][2] == ("state", "lived")
    assert rec.of("self_state.perceive")[0][2] == {"bias": 2}
    rec.remove()
    for enc in encoders.values():
        assert "forward" not in enc.__dict__ and "head_forward" not in enc.__dict__
    assert "encode" not in model.persistent_memory.__dict__ and "perceive" not in model.self_state.__dict__
    before = len(rec.calls)
    encoders["motor"](x)
    model.forward_model(x)
    model.persistent_memory.bias("s")
    assert len(rec.calls) == before


def test_encoder_input_tensors_reads_dicts_and_tensors():
    torch = pytest.importorskip("torch")
    patches, angles = torch.zeros(1, 4, 768), torch.zeros(1, 4, 4)
    got = emp.encoder_input_tensors("vision", ({"patches": patches, "angular_pos": angles, "n": 4},))
    assert set(got) == {"input/vision/patches", "input/vision/angular_pos"}
    assert got["input/vision/patches"].shape == (1, 4, 768) and got["input/vision/patches"].dtype == np.float32
    got = emp.encoder_input_tensors("proprioception", (torch.ones(1, 312, dtype=torch.float64),))
    assert set(got) == {"input/proprioception"} and got["input/proprioception"].dtype == np.float32
    assert emp.encoder_input_tensors("audio", ()) == {}


def test_dropout_values_collects_every_dropout_key_with_its_path():
    cfg = {"encoder_defaults": {"dropout": 0.0}, "modalities": {"vision": {"encoder": {"dropout": 0.1, "type": "swin"}}},
           "model": {"forward_model": {"dropout": 0.0}}, "training": {"lr": 1e-4}}
    assert emp.dropout_values(cfg) == {"encoder_defaults.dropout": 0.0, "modalities.vision.encoder.dropout": 0.1,
                                       "model.forward_model.dropout": 0.0}


def test_fixture_round_trip_keeps_tensors_and_metadata(tmp_path):
    pytest.importorskip("safetensors")
    tensors = _good_tensors()
    tensors["token/vision"][:3] = [0.5, -1.25, 3.0]
    path = str(tmp_path / "step_000000001.safetensors")
    n = emp.write_fixture(path, tensors, {"format": emp.FORMAT, "step": "1"})
    assert n == os.path.getsize(path) > 0
    back, meta = emp.read_fixture(path)
    assert meta == {"format": emp.FORMAT, "step": "1"} and set(back) == set(tensors)
    for k in tensors:
        assert np.array_equal(back[k], tensors[k]) and back[k].dtype == np.float32
    assert emp.check_fixture(back) == []


@pytest.fixture(scope="module")
def fixture_files():
    if not os.path.isdir(emp.DEFAULT_OUT):
        pytest.skip("no %s: run python -I ../../apple/scripts/export_model_pass_fixtures.py from Source/Python first"
                    % emp.DEFAULT_OUT)
    files = sorted(f for f in os.listdir(emp.DEFAULT_OUT) if f.endswith(".safetensors"))
    if not files:
        pytest.skip("no fixtures under %s" % emp.DEFAULT_OUT)
    return [os.path.join(emp.DEFAULT_OUT, f) for f in files]


def _read(path):
    pytest.importorskip("safetensors")
    return emp.read_fixture(path)


def test_three_states_pass_the_acceptance_line(fixture_files):
    assert len(fixture_files) == 3
    names = [os.path.basename(p)[:-len(".safetensors")] for p in fixture_files]
    assert names == ["step_000492800", "step_000493674", "step_000493980"]
    for path, name in zip(fixture_files, names):
        tensors, meta = _read(path)
        assert meta["format"] == emp.FORMAT and meta["state_name"] == name == "step_%09d" % int(meta["step"])
        assert tuple(json.loads(meta["prediction_senses"])) == HEADS and tuple(json.loads(meta["senses"])) == SENSES
        assert emp.check_fixture(tensors, SENSES, HEADS) == []
        assert sum(k.startswith("token/") for k in tensors) == 6
        assert sum(k.startswith("predicted/") for k in tensors) == 5


def test_command_and_wellness_shapes_and_the_wellness_sum(fixture_files):
    for path in fixture_files:
        tensors, meta = _read(path)
        assert tensors["command/motor_out"].shape == (1, 26, 9) and tensors["command/for_action"].shape == (1, 26, 9)
        assert tensors["command/handle_state"].shape == (1, 26, 10) and tensors["command/wellness_grad"].shape == (1, 26, 9)
        assert tensors["command/confidence"].shape == (1, 1)
        channels = sorted(k for k in tensors if k.startswith("wellness/") and k not in ("wellness/total", "wellness/approximate"))
        assert channels == ["wellness/capability", "wellness/coherence", "wellness/motor_state",
                            "wellness/prediction_error", "wellness/self_touch", "wellness/tactile_comfort"]
        total = float(tensors["wellness/total"][0])
        assert abs(total - sum(float(tensors[c][0, 0]) for c in channels)) < 1e-5
        assert abs(total - float(meta["total_wellness"])) < 1e-6
        assert tensors["wellness/approximate"].shape == (1, 5)
        assert len(json.loads(meta["handle_names"])) == 26


def test_first_step_after_reset_feeds_raw_tokens_on_a_zero_context(fixture_files):
    for path in fixture_files:
        tensors, meta = _read(path)
        assert tensors["forward/context_in"].shape == (1, 762, 192) and not tensors["forward/context_in"].any()
        assert tensors["forward/latents"].shape == (1, 768, 192) and tensors["forward/last_step"].shape == (1, 6, 192)
        assert tensors["forward/context_update"].shape == (1, 6, 192) and tensors["forward/memory_bias"].shape == (1, 6, 192)
        for s in SENSES:
            assert np.array_equal(tensors["token/%s" % s], tensors["token_fed/%s" % s])
            assert tensors["encoder_output/%s" % s].shape == (1, 1, 192)
        assert tensors["input/vision/patches"].shape == (1, 1160, 768) and tensors["input/vision/angular_pos"].shape == (1, 1160, 4)
        assert tensors["input/motor"].shape == (1, 26, 19) and tensors["input/proprioception"].shape == (1, 312)
        assert tensors["input/vestibular"].shape == (1, 28) and tensors["input/audio/spectral"].shape == (1, 256, 2)
        assert tensors["vestibular_norm/mean"].shape == (28,) and 0.0 < float(tensors["vestibular_norm/weight"][0]) <= 1.0
        calls = json.loads(meta["module_calls"])
        assert calls["forward_model"] == 1 and calls["prediction_heads"] == 1 and calls["motor_controller"] == 1
        assert calls["persistent_memory.bias"] >= 1 and json.loads(meta["source_mask"])["vision"] == "real"
        assert meta["device"] == "cpu" and meta["modules_eval_mode"] == "True"


def test_states_are_the_wi14_states_and_the_weights_are_the_released_file(fixture_files):
    phases = []
    for path in fixture_files:
        _t, meta = _read(path)
        phases.append((meta["caregiver_present"], meta["caregiver_phase"]))
        assert float(meta["object_pose_max_difference_m"]) <= 1e-6
        assert meta["checkpoint"] == "weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025"
        weights = os.path.join(emp.CHECKPOINT, "model.safetensors")
        if os.path.isfile(weights) and os.path.getsize(weights) > 1_000_000:
            assert meta["model_safetensors_sha256"] == emp.sha256_file(weights)
    assert phases == [("False", "absent"), ("True", "presenting"), ("True", "engaged")]
