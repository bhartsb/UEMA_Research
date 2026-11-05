# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The world-to-trainer pipe payload holds numpy arrays and plain Python values only.

PoolSource (datasource.py) starts one process per world with multiprocessing "spawn" and
sends each world's payload, the dict _raw_to_payload builds from a RawStep, through a Pipe
to the trainer. A torch.Tensor inside that payload would travel through torch's tensor
sharing instead of plain pickling, and under Python 3.12+ spawn on macOS that path hangs at
exit (PyTorch issue 153050). So the payload's leaves are checked by type here: numpy
arrays, numpy scalars, bytes, str, int, float, bool, None, and dicts, lists and tuples of
those. The other direction is checked too: PoolSource.step converts the command arrays it
sends to the workers with np.asarray before conn.send, so torch tensors handed to step()
leave the trainer as numpy arrays. The RawStep is synthetic (24 bones, 53 tactile patches,
20 collisions, 64x64 eye images); no world is built, so the test runs beside anything.
"""
import pickle

import numpy as np
import torch

from mimo_bridge import datasource as DS
from mimo_bridge.env import RawStep
from mimo_bridge.proprio_adapter import ProprioReadout
from mimo_bridge.tests.test_frame_roundtrip import make_inputs

_LEAF_TYPES = (np.ndarray, np.generic, bytes, str, int, float, bool, type(None))

# The 22 keys _raw_to_payload builds (datasource.py, the dict(...) in _raw_to_payload).
PAYLOAD_KEYS = {
    "frame_bytes", "eye_left", "eye_right", "extended_handles", "fovea_left", "fovea_right",
    "vision_layout", "sim_time", "dt", "frame", "n_contacts", "patches", "bones", "vestibular",
    "eye_angles", "gaze_deg", "ctrl", "info", "touch_pos", "touch_self", "touch_contact",
    "touch_sensitivity",
}


class _StubEnv:
    """The two attributes _raw_to_payload reads from a BridgeEnv."""

    vision_layout = [("wide", 64, 60.0)]
    n_handles = DS.N_HANDLES


def make_raw_step(seed: int = 0) -> RawStep:
    """A RawStep built from the frame-roundtrip fixture's 24 bones, 53 patches and 20
    collisions, plus 64x64 uint8 eye images, zero vestibular and touch geometry rows."""
    bones, hashes, handles, energy, body_sets, patches, cols = make_inputs(seed)
    rng = np.random.default_rng(seed)
    eye = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    n_patches = patches.shape[0]
    return RawStep(
        frame=7, sim_time=0.35, dt=0.05, patches=patches, collisions=cols,
        proprio=ProprioReadout(bones=bones, hashes=hashes, energy=np.asarray(energy, np.float32),
                               body_sets=body_sets),
        vestibular=np.zeros(6, np.float32), eye_left=eye, eye_right=eye.copy(),
        eye_angles=((0.1, -0.05), (-0.1, -0.05)), gaze_az_el_deg=(5.0, -3.0),
        n_contacts=len(cols), ctrl=np.zeros(90, np.float32), info={"step_ms": 12.5},
        touch_pos=np.zeros((n_patches, 3), np.float32), touch_self=np.zeros((n_patches, 2), np.float32),
        touch_contact=np.zeros((n_patches, 5), np.float32), touch_sensitivity=np.ones((n_patches, 2), np.float32),
    )


def leaves(value, path="payload"):
    """Every (path, leaf) under `value`, descending through dicts, lists and tuples."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, f"{path}[{k!r}]")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def _assert_plain_leaves(value, path):
    tensors = [p for p, v in leaves(value, path) if isinstance(v, torch.Tensor)]
    assert tensors == [], f"torch.Tensor on the pipe at {tensors}"
    odd = [(p, type(v).__name__) for p, v in leaves(value, path) if not isinstance(v, _LEAF_TYPES)]
    assert odd == [], f"leaves that are not numpy or plain Python: {odd}"


def test_payload_holds_no_torch_tensor_and_only_plain_leaves():
    payload = DS._raw_to_payload(_StubEnv(), make_raw_step(), DS._identity_handles(), wall_clock=1.5)
    assert isinstance(payload, dict) and set(payload) == PAYLOAD_KEYS
    _assert_plain_leaves(payload, "payload")
    assert isinstance(payload["frame_bytes"], bytes) and len(payload["frame_bytes"]) == 31232
    assert payload["eye_left"].dtype == np.uint8 and payload["eye_left"].shape == (64, 64, 3)


def test_payload_pickles_like_a_pipe_send():
    """multiprocessing.Pipe sends by pickle; the payload survives a round trip unchanged."""
    payload = DS._raw_to_payload(_StubEnv(), make_raw_step(), DS._identity_handles(), wall_clock=1.5)
    back = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert back["frame_bytes"] == payload["frame_bytes"]
    np.testing.assert_array_equal(back["bones"], payload["bones"])
    np.testing.assert_array_equal(back["eye_right"], payload["eye_right"])
    assert back["frame"] == 7 and back["vision_layout"] == (("wide", 64, 60.0),)


def test_leaf_walk_detects_a_tensor():
    """Control for the checker: a tensor planted in the step's info dict is found."""
    raw = make_raw_step()
    raw.info["planted"] = torch.zeros(2)
    payload = DS._raw_to_payload(_StubEnv(), raw, DS._identity_handles(), wall_clock=1.5)
    found = [p for p, v in leaves(payload) if isinstance(v, torch.Tensor)]
    assert found == ["payload['info']['planted']"]


class _RecordingConn:
    """Stands in for one end of a multiprocessing Pipe: keeps what send() was given."""

    def __init__(self):
        self.sent = []

    def send(self, obj):
        self.sent.append(obj)


def test_pool_step_sends_numpy_commands_to_the_workers(monkeypatch):
    """The trainer-to-worker direction: PoolSource.step given torch tensors for the command
    (26 handles x 9 numbers), gaze (2), ctrl (90) and hand_joint_goal (20) sends numpy arrays
    of those shapes, float64, and no torch.Tensor. The pool is built without workers (two
    recording pipe ends, _collect replaced), so no world process starts."""
    src = DS.PoolSource.__new__(DS.PoolSource)
    src.n_envs = 2
    src.conns = [_RecordingConn(), _RecordingConn()]
    src.procs = []
    monkeypatch.setattr(src, "_collect", lambda: [], raising=False)
    src.step(torch.ones(2, DS.N_HANDLES, 9), torch.zeros(2, 2), [True] * DS.N_HANDLES,
             ctrl=torch.zeros(2, 90), hand_joint_goal=torch.zeros(2, 20), live={"frame": "x.png"})
    for i, conn in enumerate(src.conns):
        assert len(conn.sent) == 1
        cmd, arg = conn.sent[0]
        assert cmd == "step"
        _assert_plain_leaves(arg, f"worker{i}")
        motor, gaze, mask, ctrl, render, video_control, live, hand_joint_goal, observe = arg
        assert motor.shape == (DS.N_HANDLES, 9) and motor.dtype == np.float64
        assert gaze.shape == (2,) and ctrl.shape == (90,) and hand_joint_goal.shape == (20,)
        assert mask == [True] * DS.N_HANDLES and render is None and observe is None
    assert src.conns[0].sent[0][1][6] == {"frame": "x.png"} and src.conns[1].sent[0][1][6] is None
