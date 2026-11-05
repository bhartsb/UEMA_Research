# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Export the released model's acting pass on the WI-14 states as fixtures for the Swift model port
(WI-34; REQ-B-9 encoders, REQ-B-10 forward model, REQ-B-11 weights; the references of WI-35, WI-36 and WI-39).

INPUT. (1) The released run folder weights/b1_continue_B1K_skip13_492800/run: its args.json (the trainer's 225
settings of the run that wrote the checkpoint), its checkpoints/ckpt_step_494025 (model.safetensors, 73,119,988
bytes, 18,131,745 parameters; model_receipt.json; runtime_state.pt, which carries the vestibular normaliser's
running mean and spread), its replay_training_stats.jsonl (the developmental limits the checkpoint trained under:
rom_limit and strength_limit of the "mimo" record nearest model step 494,025, read by
mimo_bridge.tools.causal_test._limits_for_run), and its observer/world/b0 recording (the same header and chunk files
export_eye_fixtures.py reads). (2) The WI-14 states as "chunk:row" pairs (--states, default 0:0,8:74,11:80 = model
steps 492,800, 493,674 and 493,980: the adult absent, presenting and engaged).

OPERATION. (1) Build the trainer's own argument namespace for a frozen probe of the checkpoint
(mimo_bridge.tools.causal_test._probe_configuration: train_mimo.parse_args defaults overwritten by the saved
args.json, one body, the device named by --device, the planner off with the planner-only flags off and
wellness_rank "raw", which ranks the goal ring and does not enter the wellness sum: lem_model.py line
`total_wellness = torch.cat(list(wellness_out["grounded"].values()), dim=-1).sum()`). (2) Build the model through
train_mimo.build_learner (the same builder the trainer and every probe tool use), load the checkpoint with
load_optimizer False, refuse if any component was re-initialised instead of loaded, put every torch module in eval
mode (no dropout, no batch statistics move; the device port runs inference only, REQ-B-17), reset the transition
state and the context (context_buffer = zeros [1, 127 x 6, 192]: ctx_len 128 minus one, times 6 senses). (3)
Build the released world exactly as export_eye_fixtures.py does (its env_kwargs_from_args and its model check
against the recording's structure fingerprint), through mimo_bridge.datasource.SingleEnvSource so the frame is
the trainer's BridgeFrame; reset it once (the trainer's own start: the body settles for 100 physics steps), then
for each state write the record's qpos, zero qvel, the mocap poses, the object flags and the mounted objects'
root poses (export_eye_fixtures.apply_object_flags, place_mounted_objects, which runs mujoco.mj_forward), restart
the world's clock at the record's simulated time (env.restart_clock: the touch state and the virtual skin start at
that instant), mujoco.mj_forward again, then collect one RawStep without a physics step
(env._collect(force_render=True): the contacts and the [53, 10] tactile fields (touch_adapter.py: 53 skin bodies
x 10 numbers), proprioception from mj_objectVelocity so zero qvel gives zero velocity, the vestibular vector, the
five camera renders, the crib audio) and turn it into the BridgeFrame the model reads
(datasource._raw_to_payload, _payload_to_frame; the motor echo is the identity echo a reset gives). (4) Run one
model step with train False, no executed command (the first model step after a reset: no delayed lesson, no
top-down gate blend, no planner) while forward hooks on the encoders, the shared output LayerNorm, the forward
model, the prediction heads, the motor controller, the wellness evaluators and the side modules the acting path
may read (persistent memory, self state, motor asked change, heads all-slot path, liveness repair) record every
call's input and output tensors. The forward model's live call is the one whose context argument is the tensor
the model kept as last_context. (5) Write one safetensors file per state with the tensors listed under OUTPUT and
a metadata block, then read every file back and check the acceptance line: six 192-number tokens, six predicted
tokens, every tensor finite.

OUTPUT. apple/UEMAKit/Tests/UEMAModelTests/Fixtures/model_pass/<state>.safetensors with <state> =
step_<model step, 9 digits>, every tensor float32 unless named, batch dimension kept where the model keeps it:
  input/<sense>/...           the encoder's input as the encoder received it: vision/patches [1, 1160, 768] (pixel
                              / 255), vision/angular_pos [1, 1160, 4]; tactile [1, 32, 19] (the 32-contact point
                              cloud); audio/spectral [1, 256, 2], audio/spatial [1, 2]; proprioception [1, 312];
                              motor [1, 26, 19] (10 achieved + 9 command columns, the command zeros: not yet known);
                              vestibular [1, 28] (centred and scaled by vestibular_norm/mean and /spread)
  encoder_features/<sense>    the encoder's per-element features before its pooling head, where the encoder has
                              one: vision [1, 1160, 192], tactile [1, 32, 192], motor [1, 26, 192]
  encoder_output/<sense>      the encoder's own output before the shared LayerNorm [1, 1, 192]
  token/<sense>               the encoder token the model keeps as last_tokens (raw, after the shared LayerNorm) [192]
  token_fed/<sense>           the token the forward model consumed (equal to token/<sense> on this first step) [192]
  forward/context_in          the context buffer the forward model read [1, 762, 192] (zeros after a reset)
  forward/memory_bias         the persistent memory's bias handed to the forward model [1, 6, 192]
                              (mimo_bridge/memory_integration.py observe: the memory's encode and bias methods,
                              recorded as persistent_memory.encode / .bias in the metadata's call counts)
  forward/latents             [1, 768, 192]; forward/last_step [1, 6, 192]; forward/context_update [1, 6, 192]
  predicted/<sense>           the prediction heads' output per sense with a head, the predicted next token [192]:
                              vision, tactile, audio, proprioception, vestibular (model_receipt.json
                              prediction_head_modalities; the motor sense has no prediction head, so a fixture holds
                              five predicted tokens, not six)
  command/handle_state        the motor controller's achieved-handle input [1, 26, 10]
  command/wellness_grad       d(wellness sum)/d(centred handle pose) handed to the controller [1, 26, 9]
  command/motor_out           the motor controller's command, 26 body handles x 9 numbers [1, 26, 9]
  command/confidence          [1, 1]
  command/for_action          [1, 26, 9]: res.motor_for_action, the command the body receives, positions in world
                              centimetres (lem_model.py step, from the line `mfa = self.execution_transform(
                              src_cmd, dev_limits)` on): positions clamped to [-1, 1] and multiplied by rom_limit
                              x 200 cm (lem/live_test.py apply_developmental_constraints), then replaced by the
                              achieved position + the difference clamped to 40 cm x rom_limit per axis; strength
                              clamped to [0, strength_limit]; exploration noise 0.0 here, so no babble; the 44
                              hand joints projected from the command (hand_kinematics.project_world_orders),
                              blended with the previous joints configure_limbs set (the world's current angles)
                              as 0.35 x new + 0.65 x previous (args.json smooth_actions 0.35) and assembled back
                              into the wrist rows (assemble_world); the command smoother (smooth_blend) returns
                              its input unchanged on this first model step after reset_context, whose previous
                              command is None. So this tensor differs from command/motor_out by the 200 cm scale
                              and the hand assembly (measured on step_000492800: largest difference 42.45); the
                              released limits are rom_limit 1.0 and strength_limit 1.0.
  wellness/total              the wellness sum, one number [1] = the sum of the wellness channels wellness/<channel>
                              [1, 1] (capability, coherence, motor_state, prediction_error, self_touch,
                              tactile_comfort on the released model); wellness/approximate [1, 5] is the approximate
                              evaluator's prediction of five of them; the movement gauge the step reports beside
                              them is not a term of the sum and is kept in the metadata only
  vestibular_norm/mean, /spread [28], /weight [1]: the normaliser's statistics from runtime_state.pt (open
                              question 12 of the specification)
The metadata block (string to string) names the state (step, chunk, row, simulated time, caregiver phase), the
checkpoint and its model.safetensors sha256, the developmental limits, the source mask, the model's d_model,
context length, senses and prediction senses, the dropout values of the configuration, the count of calls each
hooked module and method received (which side modules the acting path read: open question 1), the encoder path
each sense took, the pe source and action source the model reported, the versions and the time written.
Printed: one line per state with the token and predicted-token counts, the command shape and the wellness sum, the
model check, the peak memory; PASS or FAIL. Exit 0 only when every file passes the acceptance line as the released
model allows it: six 192-number tokens, the five predicted tokens of the receipt's prediction heads, no NaN.

Run from D:/UEMA_Research/Source/Python after the free-memory guard (one MuJoCo body with vision plus the model
on cpu; no trainer starts):
    python -m mimo_bridge.tools.free_memory_guard --need-gb 6 && python -I ../../apple/scripts/export_model_pass_fixtures.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SOURCE_PYTHON = os.path.join(REPO_ROOT, "Source", "Python")
for p in (SOURCE_PYTHON, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import export_eye_fixtures as eef  # noqa: E402  (the WI-14 exporter: the same world, the same state rebuild)

FORMAT = "uema.model_pass_fixture.v1"
RELEASED_RUN = eef.RELEASED_RUN
CHECKPOINT = os.path.join(RELEASED_RUN, "checkpoints", "ckpt_step_494025")
DEFAULT_OUT = os.path.join(REPO_ROOT, "apple", "UEMAKit", "Tests", "UEMAModelTests", "Fixtures", "model_pass")
DEFAULT_STATES = eef.DEFAULT_STATES
D_MODEL = 192          # the released width (model_receipt.json component sizes; args.json d_model 192)
N_SENSES = 6           # vision, tactile, audio, proprioception, motor, vestibular (receipt active_modalities)
SENSES = ("vision", "tactile", "audio", "proprioception", "motor", "vestibular")
# the senses with a prediction head (model_receipt.json prediction_head_modalities; lem_model.py pred_mods =
# cfg["_prediction_head_modalities"]): five of the six, the motor sense has none
PREDICTION_SENSES = ("vision", "tactile", "audio", "proprioception", "vestibular")
# the side modules the acting path may read, by their attribute name on LemLite (lem_model.py __init__)
SIDE_MODULES = ("persistent_memory", "self_state", "motor_asked_change", "heads_all_slot_path", "liveness")
# methods of side modules the live pass calls directly (no forward hook sees them): the persistent memory's
# encode and bias (mimo_bridge/memory_integration.py observe -> the forward model's memory_bias) and the self
# state's perceive (lem_model.py step, self_state_mode 'perceive')
SIDE_METHODS = (("persistent_memory", "encode"), ("persistent_memory", "bias"), ("self_state", "perceive"))


def _f32(x) -> np.ndarray:
    """INPUT: a torch tensor or array. OPERATION: detach, move to the cpu, cast to float32, make contiguous.
    OUTPUT: a float32 numpy array (a bool or integer tensor is cast too; its kind is noted by the caller)."""
    import torch
    if torch.is_tensor(x):
        x = x.detach().to("cpu")
        if x.dtype in (torch.bool,):
            x = x.to(torch.uint8)
        x = x.float().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


class CallRecorder:
    """Records every forward call of a set of torch modules.

    INPUT: (name, module) pairs. OPERATION: a forward hook with kwargs on each module appends (name, args, kwargs,
    output) with every tensor kept as a reference (detached later by the reader; nothing is copied during the
    step). OUTPUT: calls, the list in call order; counts(), the calls per name; remove() takes the hooks off."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, tuple, dict, Any]] = []
        self._handles: list = []
        self._methods: List[Tuple[Any, str]] = []

    def attach(self, name: str, module) -> None:
        def hook(_m, args, kwargs, output, _name=name):
            self.calls.append((_name, args, dict(kwargs), output))
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def attach_method(self, name: str, obj, attr: str) -> None:
        """INPUT: a name, an object and the name of one of its bound methods. OPERATION: an instance attribute of
        the same name shadows the class method with a wrapper that records (name, args, kwargs, output) and
        returns the output; remove() deletes the attribute, so the class method is back. Needed for the encoder
        methods the grounded-residual spatial path calls directly (lem/model/spatial.py encode_spatial:
        backbone_forward and head_forward), which no forward hook sees. OUTPUT: none."""
        bound = getattr(obj, attr)

        def wrapper(*args, _name=name, _bound=bound, **kwargs):
            out = _bound(*args, **kwargs)
            self.calls.append((_name, args, dict(kwargs), out))
            return out
        setattr(obj, attr, wrapper)
        self._methods.append((obj, attr))

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, _a, _k, _o in self.calls:
            out[name] = out.get(name, 0) + 1
        return out

    def of(self, name: str) -> List[Tuple[tuple, dict, Any]]:
        return [(a, k, o) for n, a, k, o in self.calls if n == name]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for obj, attr in self._methods:
            if attr in obj.__dict__:
                delattr(obj, attr)
        self._methods = []


def probe_namespace(saved_args: Dict[str, Any], device: str, out_dir: str):
    """INPUT: the released args.json as a dict, the torch device name, a folder for anything the namespace's
    --out would write. OPERATION: causal_test._probe_configuration (the trainer's parse_args defaults overwritten
    by the saved settings; n_envs 1; the device; intention off and the planner-only flags agency_drive,
    curiosity_goal, goal_locality, planner_effort_per_body off; wellness_rank raw), the run's own seed, --out at
    out_dir. OUTPUT: the argparse namespace build_learner reads."""
    from mimo_bridge.tools.causal_test import _probe_configuration
    ns = _probe_configuration(saved_args, device, int(saved_args.get("seed", 0)), False, None)
    ns.out = out_dir
    return ns


def build_model(ns, checkpoint: str):
    """INPUT: the probe namespace and the checkpoint folder. OPERATION: train_mimo.build_learner; load_checkpoint
    with load_optimizer False (runtime_state.pt restores the vestibular normaliser and the other running
    statistics); refuse a load that re-initialised a component; every torch module to eval mode;
    reset_transition_state; reset_context(1). OUTPUT: (model, the load report dict)."""
    import torch
    from mimo_bridge.train_mimo import build_learner
    model = build_learner(ns, None)
    report = model.load_checkpoint(checkpoint, load_optimizer=False)
    if report.get("reinitialized_components"):
        raise RuntimeError("checkpoint load re-initialised %s" % (report["reinitialized_components"],))
    for name, value in list(vars(model).items()):
        if isinstance(value, torch.nn.Module):
            value.eval()
    model.reset_transition_state()
    model.reset_context(1)
    return model, report


def dropout_values(config: Dict[str, Any]) -> Dict[str, float]:
    """INPUT: the model's resolved config. OPERATION: collect every key named "dropout" with its path.
    OUTPUT: {path: value}; the fixture's metadata records them (eval mode makes them inactive)."""
    out: Dict[str, float] = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "dropout" and isinstance(v, (int, float)):
                    out[path + "." + k if path else k] = float(v)
                else:
                    walk(v, path + "." + k if path else k)
    walk(config, "")
    return out


def set_world_state(env, mujoco, header: Dict[str, Any], chunk: Dict[str, Any], row: int,
                    build_alpha: Dict[int, float], build_collide: Dict[int, Tuple[int, int]]) -> Dict[str, Any]:
    """INPUT: the reset env, the recording's header, one read chunk and its row, the build's geom appearance.
    OPERATION: steps (3) of the module docstring without the collect: qpos, zero qvel, the mocap poses, the object
    flags, the mounted objects (export_eye_fixtures.apply_object_flags, place_mounted_objects, which runs
    mj_forward), then env.restart_clock at the record's simulated time, then mj_forward again (the touch state
    restart_clock started reads the contacts of the placed world). OUTPUT: the state block (name, step,
    chunk, row, simulated time, caregiver presence and phase, the mounted objects' differences)."""
    m, d = env.model, env.data
    step = int(chunk["step"][row])
    qpos = np.asarray(chunk["qpos"][row], dtype=np.float64)
    if qpos.shape != (int(m.nq),):
        raise RuntimeError("qpos %s against nq %d" % (qpos.shape, m.nq))
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    mocap = np.asarray(chunk["mocap"][row], dtype=np.float64)
    if mocap.shape != (int(m.nmocap), 7):
        raise RuntimeError("mocap %s against nmocap %d" % (mocap.shape, m.nmocap))
    d.mocap_pos[:] = mocap[:, :3]
    d.mocap_quat[:] = mocap[:, 3:]
    eef.apply_object_flags(m, header["objects"], chunk["object_flags"][row], build_alpha, build_collide)
    mounted = eef.place_mounted_objects(mujoco, m, d, header["objects"], np.asarray(chunk["object_pose"][row]))
    sim_time = float(chunk["sim_time"][row])
    env.restart_clock(sim_time)
    mujoco.mj_forward(m, d)
    phase_names = [str(x) for x in np.asarray(chunk["caregiver_phase_names"]).reshape(-1)]
    phase = int(chunk["caregiver_phase"][row])
    return dict(name="step_%09d" % step, step=step, row=int(row), sim_time_s=sim_time,
                caregiver_present=bool(chunk["caregiver_present"][row]),
                caregiver_phase=phase_names[phase] if 0 <= phase < len(phase_names) else str(phase),
                mounted_objects=mounted,
                object_pose_max_difference_m=max(float(np.linalg.norm(
                    np.asarray(d.xpos[int(o["root"])]) - np.asarray(chunk["object_pose"][row][k][:3])))
                    for k, o in enumerate(header["objects"])))


def frame_at_state(src):
    """INPUT: the SingleEnvSource whose env sits at the state. OPERATION: env._collect(force_render=True) (one
    RawStep from the current state, no physics step), datasource._raw_to_payload with the source's current motor
    echo (the identity echo after a reset), datasource._payload_to_frame. OUTPUT: the BridgeFrame."""
    from mimo_bridge.datasource import _payload_to_frame, _raw_to_payload
    raw = src.env._collect(force_render=True)
    payload = _raw_to_payload(src.env, raw, src._echo, 0.0)
    return _payload_to_frame(payload, 0)


def configure_limbs(model, src, frame, parent_relative_targets: str) -> Dict[str, Any]:
    """INPUT: the model, the SingleEnvSource at the state, the state's BridgeFrame, the run's
    parent_relative_targets setting. OPERATION: what the trainer does before its first model step
    (train_mimo.py, the block `if model.hand_target_mode != 'world'` and `planner_calf_geometry`): with a
    wrist-relative hand target the world's hand state (datasource._hand_state: the hand joints' geometry, their
    current angles and velocities from qpos and qvel) is written into frame.meta as hand_joint_angles and
    hand_joint_velocities, and model.configure_hand_geometry receives the geometry with the current angles as
    the previous joints (held goal when the world holds one); the calf projection is set from the frame's
    metadata (None at the released setting "off"). OUTPUT: a facts dict (hand joint count, calf projection)."""
    from mimo_bridge.train_mimo import planner_calf_geometry
    facts: Dict[str, Any] = dict(hand_joints=0, calf_projection=False)
    if model.hand_target_mode != "world":
        states = src.hand_state()
        if not states or states[0] is None:
            raise RuntimeError("the model expects a wrist-relative hand target and the world has no hand-target control")
        hs = states[0]
        frame.meta["hand_joint_angles"] = hs["current"]
        frame.meta["hand_joint_velocities"] = hs["velocity"]
        initial = np.stack([hs["current"] if hs["held"] is None else hs["held"]])
        model.configure_hand_geometry(hs["geometry"], initial if model._hand_previous_joints is None else None)
        facts["hand_joints"] = int(np.asarray(hs["current"]).shape[0])
    calf = planner_calf_geometry([frame.meta], parent_relative_targets)
    model.configure_calf_projection(calf)
    facts["calf_projection"] = calf is not None
    return facts


def hook_model(model) -> Tuple[CallRecorder, Dict[str, Any]]:
    """INPUT: the LemLite. OPERATION: one CallRecorder; its method shadows (attach_method) on each encoder's
    forward, backbone_forward and head_forward where the encoder has them, and on the side methods SIDE_METHODS;
    its forward hooks (attach) on the registry's output LayerNorm, the forward model, the prediction heads, the
    motor controller, the wellness evaluators and each present side module (SIDE_MODULES) that is a torch
    module. OUTPUT: (recorder, {name: module} for the hooked modules; the shadowed methods are not in it)."""
    import torch
    rec = CallRecorder()
    modules: Dict[str, Any] = {}
    for sense, enc in model.encoder_registry.encoders.items():
        # the registry's own path calls the module (forward); the grounded-residual spatial path calls
        # backbone_forward then head_forward on vision and tactile (lem/model/spatial.py encode_spatial)
        for attr in ("forward", "backbone_forward", "head_forward"):
            if callable(getattr(enc, attr, None)):
                rec.attach_method("encoder.%s.%s" % (sense, attr), enc, attr)
    if model.encoder_registry.output_norm is not None:
        modules["encoder_registry.output_norm"] = model.encoder_registry.output_norm
    modules["forward_model"] = model.forward_model
    modules["prediction_heads"] = model.prediction_heads
    modules["motor_controller"] = model.motor_controller
    modules["wellness"] = model.wellness
    for name in SIDE_MODULES:
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Module):
            modules[name] = mod
    for name, mod in modules.items():
        rec.attach(name, mod)
    for owner, attr in SIDE_METHODS:
        mod = getattr(model, owner, None)
        if mod is not None and callable(getattr(mod, attr, None)):
            rec.attach_method("%s.%s" % (owner, attr), mod, attr)
    return rec, modules


def encoder_input_tensors(sense: str, args: tuple) -> Dict[str, np.ndarray]:
    """INPUT: a sense and the positional arguments its encoder module received. OPERATION: a dict argument
    (vision: patches and angular_pos; audio: spectral and spatial) gives one tensor per key; a tensor argument
    gives one. OUTPUT: {"input/<sense>" or "input/<sense>/<key>": float32 array}."""
    import torch
    out: Dict[str, np.ndarray] = {}
    if not args:
        return out
    first = args[0]
    if isinstance(first, dict):
        for k, v in first.items():
            if torch.is_tensor(v):
                out["input/%s/%s" % (sense, k)] = _f32(v)
    elif torch.is_tensor(first):
        out["input/%s" % sense] = _f32(first)
    return out


def run_pass(model, frame, dev: Dict[str, Any], step_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Any]:
    """INPUT: the model after reset_context(1), one BridgeFrame, the developmental limits, the model step index.
    OPERATION: step (4) of the module docstring: hooks on, model.step([frame], dev, train=False, step_index,
    executed_action None), hooks off; the live forward-model call is the one whose context argument is
    model.last_context; the tensors named under OUTPUT are gathered. OUTPUT: (tensors, facts as strings, the
    StepResult)."""
    import torch
    rec, modules = hook_model(model)
    try:
        # no torch.no_grad here: the step differentiates the wellness sum with respect to the centred handle
        # pose (wellness_grad) before the motor controller runs; the weights stay unchanged (train=False)
        res = model.step([frame], dict(dev, exploration_noise=0.0), train=False, step_index=step_index,
                         executed_action=None, executed_order_id=None)
    finally:
        rec.remove()
    tensors: Dict[str, np.ndarray] = {}
    facts: Dict[str, str] = {}
    # encoder inputs and the modules' own outputs, from whichever path ran
    encoder_path: Dict[str, str] = {}
    for sense in model.active:
        fwd = rec.of("encoder.%s.forward" % sense)
        backbone = rec.of("encoder.%s.backbone_forward" % sense)
        head = rec.of("encoder.%s.head_forward" % sense)
        if len(fwd) == 1:
            # the module's own forward ran (an encoder whose forward is backbone_forward then head_forward
            # shows those two calls as well: the motor encoder); the features are kept when one backbone call ran
            args, _kw, out = fwd[0]
            if len(backbone) == 1:
                tensors["encoder_features/%s" % sense] = _f32(backbone[0][2])
            if sense == "vision" and len(args) >= 2:
                tensors["input/vision/patches"] = _f32(args[0])
                tensors["input/vision/angular_pos"] = _f32(args[1])
            elif sense == "audio" and len(args) >= 2:
                tensors["input/audio/spectral"] = _f32(args[0])
                tensors["input/audio/spatial"] = _f32(args[1])
            else:
                tensors.update(encoder_input_tensors(sense, args))
            tensors["encoder_output/%s" % sense] = _f32(out)
            encoder_path[sense] = "forward"
        elif len(backbone) == 1 and len(head) == 1 and not fwd:
            bargs, _bk, features = backbone[0]
            _hargs, _hk, out = head[0]
            if sense == "vision" and len(bargs) >= 2:
                tensors["input/vision/patches"] = _f32(bargs[0])
                tensors["input/vision/angular_pos"] = _f32(bargs[1])
            else:
                tensors.update(encoder_input_tensors(sense, bargs))
            tensors["encoder_features/%s" % sense] = _f32(features)
            tensors["encoder_output/%s" % sense] = _f32(out)
            encoder_path[sense] = "backbone_forward+head_forward"
        else:
            raise RuntimeError("encoder %s: forward %d, backbone_forward %d, head_forward %d calls; one live path "
                               "expected" % (sense, len(fwd), len(backbone), len(head)))
    facts["encoder_path"] = json.dumps(encoder_path, sort_keys=True)
    for sense, tok in model.last_tokens.items():
        tensors["token/%s" % sense] = _f32(tok).reshape(-1)
    for sense, tok in model.last_tokens_fed.items():
        tensors["token_fed/%s" % sense] = _f32(tok).reshape(-1)
    # the forward model's live call
    live = [(a, k, o) for a, k, o in rec.of("forward_model")
            if len(a) >= 2 and torch.is_tensor(a[1]) and a[1] is model.last_context]
    if len(live) != 1:
        raise RuntimeError("%d forward-model calls read the model's last_context; one expected (%d calls in all)"
                           % (len(live), len(rec.of("forward_model"))))
    fargs, fkw, fout = live[0]
    tensors["forward/context_in"] = _f32(fargs[1])
    if fkw.get("memory_bias") is not None:
        tensors["forward/memory_bias"] = _f32(fkw["memory_bias"])
    latents, last_step, ctx_update = fout[0], fout[1], fout[2]
    tensors["forward/latents"] = _f32(latents)
    tensors["forward/last_step"] = _f32(last_step)
    tensors["forward/context_update"] = _f32(ctx_update)
    for sense, pred in model.last_predictions.items():
        tensors["predicted/%s" % sense] = _f32(pred).reshape(-1)
    # the motor controller's live call: its latents are the live forward call's latents
    mc = [(a, k, o) for a, k, o in rec.of("motor_controller") if len(a) >= 1 and a[0] is latents]
    if len(mc) != 1:
        raise RuntimeError("%d motor-controller calls read the live latents; one expected" % len(mc))
    margs, mkw, mout = mc[0]
    tensors["command/handle_state"] = _f32(margs[1])
    wg = mkw.get("wellness_grad", margs[2] if len(margs) > 2 else None)
    if wg is not None:
        tensors["command/wellness_grad"] = _f32(wg)
    tensors["command/motor_out"] = _f32(mout[0])
    tensors["command/confidence"] = _f32(mout[1])
    tensors["command/for_action"] = _f32(res.motor_for_action)
    tensors["wellness/total"] = np.asarray([float(res.total_wellness)], dtype=np.float32)
    for channel, value in model.last_wellness_grounded.items():
        tensors["wellness/%s" % channel] = _f32(value)
    wl = [(a, k, o) for a, k, o in rec.of("wellness") if len(a) >= 3 and a[2] is latents]
    if len(wl) == 1 and isinstance(wl[0][2], dict) and wl[0][2].get("approximate") is not None:
        tensors["wellness/approximate"] = _f32(wl[0][2]["approximate"])
    norm = getattr(model, "vestibular_norm", None)
    if norm is not None:
        tensors["vestibular_norm/mean"] = _f32(norm.mean)
        tensors["vestibular_norm/spread"] = _f32(norm.spread())
        tensors["vestibular_norm/weight"] = np.asarray([float(norm.weight)], dtype=np.float32)
        facts["vestibular_norm_n_real"] = str(int(norm.n_real))
    facts["module_calls"] = json.dumps(rec.counts(), sort_keys=True)
    facts["side_modules_present"] = json.dumps({n: (n in modules) for n in SIDE_MODULES}, sort_keys=True)
    facts["source_mask"] = json.dumps(dict(res.source_mask), sort_keys=True)
    facts["pe_source"] = str(getattr(model, "last_pe_source", ""))
    facts["action_source"] = str(getattr(model, "last_action_source", ""))
    facts["wellness_channels"] = json.dumps({k: float(v) for k, v in res.wellness.items()}, sort_keys=True)
    facts["total_wellness"] = repr(float(res.total_wellness))
    facts["motor_confidence"] = repr(float(res.motor_confidence))
    facts["mean_pe"] = repr(float(res.mean_pe))
    return tensors, facts, res


def check_fixture(tensors: Dict[str, np.ndarray], senses: Sequence[str] = SENSES,
                  prediction_senses: Sequence[str] = PREDICTION_SENSES) -> List[str]:
    """INPUT: the tensors of one fixture (name -> array), the six senses, the senses with a prediction head.
    OPERATION: the acceptance line as the released model allows it: token/<sense> present for the six senses with
    192 numbers each; predicted/<sense> present for every prediction sense with 192 numbers each and for no other
    sense; every tensor finite (no NaN, no inf). OUTPUT: the list of failures (empty = pass)."""
    bad: List[str] = []
    for sense in senses:
        key = "token/%s" % sense
        if key not in tensors:
            bad.append("%s missing" % key)
        elif tuple(np.asarray(tensors[key]).shape) != (D_MODEL,):
            bad.append("%s has shape %s, not (%d,)" % (key, np.asarray(tensors[key]).shape, D_MODEL))
    for sense in prediction_senses:
        key = "predicted/%s" % sense
        if key not in tensors:
            bad.append("%s missing" % key)
        elif tuple(np.asarray(tensors[key]).shape) != (D_MODEL,):
            bad.append("%s has shape %s, not (%d,)" % (key, np.asarray(tensors[key]).shape, D_MODEL))
    extra = sorted(k for k in tensors if k.startswith("predicted/") and k.split("/", 1)[1] not in prediction_senses)
    if extra:
        bad.append("predicted tokens for senses without a prediction head: %s" % extra)
    for key, arr in tensors.items():
        if not np.isfinite(np.asarray(arr)).all():
            bad.append("%s holds a non-finite number" % key)
    return bad


def sha256_file(path: str) -> str:
    """INPUT: a path. OPERATION: sha256 over the file's bytes in 1 MiB pieces. OUTPUT: the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def write_fixture(path: str, tensors: Dict[str, np.ndarray], metadata: Dict[str, str]) -> int:
    """INPUT: the output path, the tensors, the metadata (every value a string). OPERATION: safetensors.numpy
    save_file. OUTPUT: the file's byte count."""
    from safetensors.numpy import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, path, metadata=metadata)
    return os.path.getsize(path)


def read_fixture(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """INPUT: a fixture path. OPERATION: safetensors safe_open with framework "np". OUTPUT: (tensors, metadata)."""
    from safetensors import safe_open
    tensors: Dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release-run", default=RELEASED_RUN, help="the released run folder (args.json, observer/)")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="the checkpoint folder (model.safetensors, receipt)")
    parser.add_argument("--body", type=int, default=0, help="the observer body folder b<body> (default 0)")
    parser.add_argument("--states", default=DEFAULT_STATES, help="chunk:row pairs, comma separated (the WI-14 states)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="the fixture folder (default: the UEMAModelTests folder)")
    parser.add_argument("--device", default="cpu", help="the torch device the model runs on (cpu: the reference)")
    parser.add_argument("--scene-dir", default="", help="scene folder for the build (default: a temporary folder, "
                                                         "removed afterwards)")
    args = parser.parse_args(argv)
    import mujoco
    import torch
    import mimo_bridge  # noqa: F401  (installs repo_paths: the saved command's development-machine paths read here)
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.observer_record import read_chunk
    from mimo_bridge.tools.causal_test import _limits_for_run
    states = eef.parse_states(args.states)
    with open(os.path.join(args.release_run, "args.json"), encoding="utf-8") as f:
        args_json = json.load(f)
    body_dir = os.path.join(args.release_run, "observer", "world", "b%d" % args.body)
    chunk_files = sorted(glob.glob(os.path.join(body_dir, "chunk_*.npz")))
    if not chunk_files:
        print("REFUSED: no chunk files under %s" % body_dir)
        return 2
    for chunk_i, row in states:
        if chunk_i >= len(chunk_files):
            print("REFUSED: chunk %d asked, %d chunk files under %s" % (chunk_i, len(chunk_files), body_dir))
            return 2
    weights = os.path.join(args.checkpoint, "model.safetensors")
    if not os.path.isfile(weights) or os.path.getsize(weights) < 1_000_000:
        print("REFUSED: %s is missing or a Git LFS pointer; run fetch_weights.py" % weights)
        return 2
    written_at = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
    work = tempfile.mkdtemp(prefix="model_pass_fixtures_")
    scene_dir = args.scene_dir or os.path.join(work, "scenes")
    torch.manual_seed(int(args_json.get("seed", 0)))
    np.random.seed(int(args_json.get("seed", 0)))
    ns = probe_namespace(args_json, args.device, os.path.join(work, "unused_out"))
    model, load_report = build_model(ns, args.checkpoint)
    dev, dev_source = _limits_for_run(args.checkpoint, model)
    print("model: device %s, loaded %d components from %s; limits %s (%s)"
          % (model.device, len(load_report.get("loaded_components", []) or []), args.checkpoint,
             {k: dev[k] for k in ("phase_name", "rom_limit", "strength_limit")}, dev_source))
    kw = eef.env_kwargs_from_args(args_json, scene_dir, eef.speech_dir_here(str(args_json["caregiver_speech_dir"])),
                                  tag="_modelfix")
    src = SingleEnvSource(**kw)
    env = src.env
    summary: List[Dict[str, Any]] = []
    try:
        first = read_chunk(chunk_files[0])
        header = first["header"]
        check = eef.check_model_against_header(env, header)
        print("model check: built fingerprint %s, header %s, counts %s against %s -> %s"
              % (check["structure_fingerprint"][:16], check["header_fingerprint"][:16], check["built"],
                 check["header"], "MATCH" if check["match"] else "MISMATCH"))
        if not check["match"]:
            print("REFUSED: the built world is not the recorded world; nothing written")
            return 1
        src.reset()
        m = env.model
        build_alpha = {int(g): float(m.geom_rgba[g, 3]) for g in range(int(m.ngeom))}
        build_collide = {int(g): (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(int(m.ngeom))}
        weights_sha = sha256_file(weights)
        common = dict(
            format=FORMAT, written_by="apple/scripts/export_model_pass_fixtures.py", written_at=written_at,
            versions=json.dumps(dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                     mujoco=mujoco.__version__), sort_keys=True),
            checkpoint=os.path.relpath(args.checkpoint, REPO_ROOT).replace("\\", "/"),
            model_safetensors_sha256=weights_sha, device=str(model.device),
            dev_limits=json.dumps({k: dev[k] for k in ("phase_name", "rom_limit", "strength_limit")}, sort_keys=True),
            dev_limits_source=dev_source, d_model=str(model.d_model), context_length=str(model.ctx_len),
            senses=json.dumps(list(model.active)), prediction_senses=json.dumps(list(model.pred_mods)),
            handle_names=json.dumps(list(model.handle_names)),
            residual_prediction=str(bool(model.config["model"].get("residual_prediction", False))),
            dropout=json.dumps(dropout_values(model.config), sort_keys=True), modules_eval_mode="True",
            probe_configuration="causal_test._probe_configuration: n_envs 1, intention off, agency_drive "
                                "curiosity_goal goal_locality planner_effort_per_body off, wellness_rank raw",
            step_call="model.step([frame], dev with exploration_noise 0.0, train=False, executed_action=None): "
                      "the first model step after reset_context(1), context zeros, no delayed lesson, no gate blend",
            world_model_check=json.dumps(check, sort_keys=True),
            batch_dims="input/*, encoder_output/*, forward/*, command/*, wellness/<channel>: batch 1 kept; "
                       "token/*, predicted/*: [192]; wellness/total, vestibular_norm/weight: [1]")
        chunks: Dict[int, Dict[str, Any]] = {0: first}
        for chunk_i, row in states:
            if chunk_i not in chunks:
                chunks[chunk_i] = read_chunk(chunk_files[chunk_i], header)
            chunk = chunks[chunk_i]
            if row >= len(chunk["step"]):
                print("REFUSED: row %d asked, chunk %s holds %d records" % (row, chunk_files[chunk_i], len(chunk["step"])))
                return 2
            model.reset_transition_state()
            model.reset_context(1)
            state = set_world_state(env, mujoco, header, chunk, row, build_alpha, build_collide)
            frame = frame_at_state(src)
            limbs = configure_limbs(model, src, frame, str(args_json.get("parent_relative_targets", "off") or "off"))
            tensors, facts, res = run_pass(model, frame, dev, int(state["step"]))
            facts["limbs"] = json.dumps(limbs, sort_keys=True)
            meta = dict(common, state_name=state["name"], step=str(state["step"]),
                        chunk_file=os.path.basename(chunk_files[chunk_i]), row=str(row),
                        sim_time_s=repr(state["sim_time_s"]), caregiver_present=str(state["caregiver_present"]),
                        caregiver_phase=state["caregiver_phase"],
                        object_pose_max_difference_m=repr(state["object_pose_max_difference_m"]), **facts)
            path = os.path.join(args.out, state["name"] + ".safetensors")
            nbytes = write_fixture(path, tensors, meta)
            summary.append(dict(name=state["name"], path=path, bytes=nbytes, tokens=len(
                [k for k in tensors if k.startswith("token/")]), predicted=len(
                [k for k in tensors if k.startswith("predicted/")]), command=list(tensors["command/motor_out"].shape),
                total_wellness=float(res.total_wellness), n_tensors=len(tensors), calls=json.loads(facts["module_calls"])))
    finally:
        src.close()
        # work holds the namespace's unused --out folder and, when --scene-dir is empty, the scene folder; a
        # given --scene-dir lies outside work and stays
        shutil.rmtree(work, ignore_errors=True)
    ok = True
    for s in summary:
        tensors, meta = read_fixture(s["path"])
        heads = tuple(json.loads(meta.get("prediction_senses", "[]")))
        bad = check_fixture(tensors, SENSES, heads)
        if heads != PREDICTION_SENSES:
            bad.append("the model's prediction senses %s are not the receipt's %s" % (heads, PREDICTION_SENSES))
        print("state %s: %d tokens of %d numbers, %d predicted tokens (heads %s), command %s, wellness sum %.6f, "
              "%d tensors, %s bytes, side-module calls %s -> %s" % (
                  s["name"], s["tokens"], D_MODEL, s["predicted"], list(heads), s["command"], s["total_wellness"],
                  s["n_tensors"], f"{s['bytes']:,}",
                  {k: v for k, v in s["calls"].items() if k.split(".")[0] in SIDE_MODULES}, s["path"]))
        if meta.get("format") != FORMAT:
            bad.append("format %r" % meta.get("format"))
        for b in bad:
            print("FAIL: %s: %s" % (s["name"], b))
            ok = False
    try:
        import psutil
        peak = psutil.Process().memory_info()
        print("peak working set %s MB" % f"{int(getattr(peak, 'peak_wset', peak.rss) / 2 ** 20):,}")
    except Exception as ex:   # the memory line is a report, never a gate
        print("peak working set unread: %r" % (ex,))
    if ok and len(summary) == len(states):
        print("PASS: %d states, each with six %d-number tokens and the %d predicted tokens of the receipt's prediction "
              "heads (%s; the motor sense has no head), every tensor finite"
              % (len(summary), D_MODEL, len(PREDICTION_SENSES), ", ".join(PREDICTION_SENSES)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
