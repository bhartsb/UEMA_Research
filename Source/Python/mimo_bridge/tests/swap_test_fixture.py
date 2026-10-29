# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A small H84-shaped learner driven on synthetic frames: no MuJoCo world, no checkpoint, CPU only.

Used by tests/test_swap_test_phases.py, tests/test_controllability_shadow_gate.py and by
docs/gain_of_function_v4/swap_test_phases/lockstep_vs_production.py and resume_from_production_state.py, which run
this module under production's code and under a branch's code and compare the two. It therefore uses
only what both have: train_mimo.parse_args / build_learner (the learner is built from H84's own launch flags, at a
small width), the real frame packing and tokenizer (frame_encoder.encode_frame, datasource._payload_to_frame) and
LemLite.step, driven with the trainer's latch rule (a new command latched on model steps that are multiples of
--motor-hold; the executed command is the latched one with the eyes' row written from the gaze sent).

The synthetic bodies are not physics: each of the 26 handles moves a fifth of the way from where it is toward its
commanded position each model step, plus seeded noise; the eyes move six tenths of the way toward the gaze sent;
skin contacts, vision patches, audio and vestibular readings are seeded random numbers. Their numbers are plumbing,
not calibration evidence."""
import copy
import hashlib
import json
import os

import numpy as np
import torch

# H84's launch flags (Saved/mimo_bridge/training_continuation.json, command of segment 41s resumed 2026-10-01
# 9:35 PM PDT) that act inside the learner, at a small width. Left out: what needs a MuJoCo world, a file or a
# body-function producer (--hand-target-mode wrist_relative_v1, --hand-goals memory, --self-state learn,
# --body-function modulate, --self-prediction grounded_residual, --contact-profile mechanical_v4, the persistent
# memory's repair and the motor input fold, --proprio-token-repair, the calibration files, the latent-body shadow,
# the caregiver, the observer) and what only the trainer reads.
H84_LEARNER_FLAGS = [
    "--action-repeat", "5", "--motor-hold", "5", "--intention", "on", "--intention-every", "10",
    "--next-frame-loss", "on", "--residual-prediction", "on", "--top-down-gate", "on",
    "--imagination", "on", "--imagination-steps", "3", "--plan-horizon", "earned",
    "--think-harder", "on", "--wellness-mode", "baseline", "--motor-state-formula", "boundary",
    "--body-layout", "trunk_v1", "--body-schema-updates", "on",
    "--agency-drive", "on", "--movement-drive", "on", "--movement-mref", "24.0",
    "--movement-bonus", "0.2", "--movement-penalty", "0.25",
    "--curiosity-goal", "on", "--curiosity-mode", "additive", "--curiosity-band", "0.02:0.4",
    "--consequence-w", "0.5", "--consequence-self-contact", "exclude",
    "--efference-heads", "on", "--causal-chain", "on", "--vestibular-sense", "on",
    "--wellness-rank", "normalized", "--controllability", "on", "--learning-progress", "on",
    "--contact-head", "on", "--contact-control-wellness", "0.5",
    "--motor-synergy", "on", "--synergy-planner", "off", "--synergy-babble", "off",
    "--smooth-actions", "0.35", "--smooth-source", "proposal",
    "--next-frame-norm", "persistence", "--liveness-repair", "on", "--liveness-rule", "live_line",
    "--action-tenure-loss", "on", "--action-tenure-token-reach", "on",
    "--vision-gate", "self-motion", "--audio", "real", "--audio-freq-bands", "on",
    "--placeholder-senses", "on", "--persistent-memory", "on",
]
SMALL = ["--context-length", "8", "--d-model", "32", "--core-blocks", "1", "--encoder-width", "32",
         "--encoder-layers", "1", "--vision-width", "32", "--vision-swin-layers", "1"]


def argv(n_envs=3, motor_hold=5, intention_every=10, out=None, extra=()):
    a = ["--out", out or os.path.join(os.environ.get("TEMP", "."), "swap_test_fixture_out"),
         "--device", "cpu", "--n-envs", str(n_envs), "--seed", "0"] + list(H84_LEARNER_FLAGS) + list(SMALL)
    a[a.index("--motor-hold") + 1] = str(motor_hold)
    a[a.index("--intention-every") + 1] = str(intention_every)
    return a + list(extra)


def build(args_list):
    """(learner, parsed args) from a launch line, built as the trainer builds it (train_mimo.build_learner)."""
    from mimo_bridge import train_mimo as T
    args = T.parse_args(args_list)
    args.n_envs = int(args.n_envs)
    model = T.build_learner(args, None)
    model.reset_context(args.n_envs)
    return model, args


class SyntheticWorld:
    """Seeded synthetic bodies of the trunk_v1 layout (26 handles, 53 skin patches)."""

    N_PATCHES = 53

    def __init__(self, n_envs, seed=0, vision=True, n_patches=32):
        from mimo_bridge.body_layout import names_for
        self.H = len(names_for("trunk_v1"))
        self.B = int(n_envs)
        self.rng = np.random.default_rng(seed)
        self.vision = vision
        # vision patches per body. H84: 1,160 = (256 wide + 324 fovea) x 2 eyes - the wide view 256 px (env.py's
        # eye_size default; H84 sets none), the fovea 288 px (--fovea-size), 16 px patches (768 = 16 x 16 x 3)
        self.n_patches = int(n_patches)
        self.pos = self.rng.normal(0.0, 20.0, (self.B, self.H, 3)).astype(np.float32)
        self.pos[..., 2] += 15.0
        q = self.rng.normal(0.0, 1.0, (self.B, self.H, 4)).astype(np.float32)
        self.quat = q / np.linalg.norm(q, axis=-1, keepdims=True)
        self.vel = np.zeros((self.B, self.H, 3), np.float32)
        self.eye = np.zeros((self.B, 2), np.float32)
        self.frame = 0
        self.t = 0.0
        # the random stream as it stood before the newest frames were drawn (state() saves it)
        self._rng_before_frames = None

    def state(self):
        """What a save keeps of the bodies: their arrays and counters, and the random stream as it stood before
        the newest frames were drawn, so that restore() followed by frames() gives those same frames again and
        leaves the stream where it was."""
        return dict(pos=self.pos.copy(), quat=self.quat.copy(), vel=self.vel.copy(), eye=self.eye.copy(),
                    frame=int(self.frame), t=float(self.t), rng=copy.deepcopy(self._rng_before_frames))

    def restore(self, saved):
        """Take back what state() gave (in a new process: build the bodies with the same count first)."""
        self.pos, self.quat = saved["pos"].copy(), saved["quat"].copy()
        self.vel, self.eye = saved["vel"].copy(), saved["eye"].copy()
        self.frame, self.t = int(saved["frame"]), float(saved["t"])
        self.rng.bit_generator.state = copy.deepcopy(saved["rng"])

    def _payload(self, b):
        from mimo_bridge import frame_encoder as FE
        from mimo_bridge.vestibular_adapter import VESTIBULAR_DIM
        H, P = self.H, self.N_PATCHES
        bones = np.zeros((H, 11), np.float32)
        bones[:, 0:4], bones[:, 4:7], bones[:, 7:10], bones[:, 10] = self.quat[b], self.pos[b], self.vel[b], 1.0
        handles = np.zeros((H, 10), np.float32)
        handles[:, 0:3], handles[:, 3:7], handles[:, 8], handles[:, 9] = self.pos[b], self.quat[b], 0.5, 1.0
        patches = np.zeros((P, 10), np.float32)
        touched = self.rng.random(P) < 0.15
        k = int(touched.sum())
        patches[touched, 0] = self.rng.random(k) * 500.0
        patches[touched, 1:8] = self.rng.normal(0, 0.1, (k, 7))
        buf = FE.encode_frame(frame_number=self.frame, timestamp=self.t, dt=0.05, wall_clock=0.0,
                              gaze_az_deg=float(np.degrees(self.eye[b, 0])),
                              gaze_el_deg=float(np.degrees(self.eye[b, 1])),
                              bones=bones[:24], bone_hashes=None, handles=handles[:24], energy=(0.0, 0.0, 0.0),
                              body_sets=None, patches=patches, collisions=[], vision_valid=False,
                              audio_spectral=self.rng.random((256, 2)).astype(np.float32) * 0.1,
                              audio_meta={"ild": 0.0, "itd": 0.0})
        vest = np.zeros(VESTIBULAR_DIM, np.float32)
        vest[:-1] = self.rng.normal(0, 0.5, VESTIBULAR_DIM - 1)
        vest[-1] = 1.0
        state = (self.rng.random(P) < 0.1).astype(int) * self.rng.integers(1, 4, P)
        task = {"hand_in_view": bool(self.rng.random() < 0.3), "hand_in_fovea": bool(self.rng.random() < 0.1),
                "contact_patch_state": [int(s) for s in state], "self_contact_onsets": [],
                "contact_onsets_world": 0, "mobile_speed_cms": 0.0}
        ea = ((float(self.eye[b, 0]), float(self.eye[b, 1])),) * 2
        return dict(frame_bytes=bytes(buf), eye_left=None, eye_right=None, extended_handles=handles,
                    fovea_left=None, fovea_right=None, vision_layout=None, sim_time=self.t, dt=0.05,
                    frame=self.frame, n_contacts=k, patches=patches, bones=bones, vestibular=vest,
                    eye_angles=ea, gaze_deg=tuple(np.degrees(self.eye[b]).tolist()),
                    ctrl=np.zeros(4, np.float32), info={"task": task},
                    touch_pos=self.rng.normal(0, 0.1, (P, 3)).astype(np.float32),
                    touch_self=np.zeros((P, 2), np.float32), touch_contact=None, touch_sensitivity=None)

    def frames(self):
        from mimo_bridge.datasource import _payload_to_frame
        self._rng_before_frames = copy.deepcopy(self.rng.bit_generator.state)
        out = [_payload_to_frame(self._payload(b), b) for b in range(self.B)]
        if self.vision:
            n = self.n_patches
            for b, f in enumerate(out):
                f.vision_patches = self.rng.random((n, 768)).astype(np.float32)
                ang = np.zeros((n, 4), np.float32)
                ang[:, 0] = np.linspace(-0.5, 0.5, n) + self.eye[b, 0]
                ang[:, 1] = self.eye[b, 1]
                ang[:, 2] = (np.arange(n) >= n // 2).astype(np.float32)
                ang[:, 3] = 0.1
                f.vision_angles = ang
        return out

    def step(self, send_motor, gaze):
        cmd = np.asarray(send_motor, np.float32)
        old = self.pos.copy()
        self.pos = (self.pos + 0.2 * (np.clip(cmd[..., 0:3], -150, 150) - self.pos) * (cmd[..., 8:9] > 0)
                    + self.rng.normal(0, 0.3, self.pos.shape).astype(np.float32))
        q = self.quat + 0.2 * cmd[..., 3:7] + self.rng.normal(0, 0.02, self.quat.shape).astype(np.float32)
        self.quat = q / np.linalg.norm(q, axis=-1, keepdims=True)
        self.vel = (self.pos - old) / 0.05
        if gaze is not None:
            g = np.asarray(gaze, np.float32).reshape(self.B, 2)
            self.eye = self.eye + 0.6 * (g - self.eye)
        self.frame += 5
        self.t += 0.05
        return self.frames()


def _bytes(t):
    t = t.detach().cpu().contiguous()
    return (t.float() if t.dtype == torch.bfloat16 else t).numpy().tobytes()


def sha(x):
    if torch.is_tensor(x):
        return hashlib.sha256(_bytes(x)).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16]


def state_digest(model):
    """One hash over every parameter and buffer (persistent or not: the running centres too) of every learned
    component, every Adam moment and the planner's goal ring values."""
    from lem.model.efference import _named_components
    h = hashlib.sha256()
    for comp, mod in sorted(_named_components(model).items()):
        for n, p in list(mod.named_parameters()) + list(mod.named_buffers()):
            h.update((comp + "." + n).encode())
            h.update(_bytes(p))
    for g in model.optimizer.param_groups:
        for p in g["params"]:
            for k, v in sorted(model.optimizer.state.get(p, {}).items()):
                if torch.is_tensor(v):
                    h.update(k.encode())
                    h.update(_bytes(v))
    for _, w in list(getattr(model, "goal_ring", []) or []):
        h.update(_bytes(w))
    return h.hexdigest()[:16]


def drive(model, world, n_steps, hold=5, start=0, babble=0.05, digest_every=10, on_step=None, state=None):
    """Run n_steps model steps with the trainer's latch rule; one row per model step of what training produced:
    the loss, the gradient norm, the command and gaze sent, the loss breakdown, the wellness values, the three
    random streams, and every digest_every model steps the state digest.
    state: None, or a dict. A dict that holds a previous call's loop state (the frames the bodies last gave, the
    held command, the previous executed command with its number and gaze, the count of latched commands) makes
    this call continue from it instead of starting; any dict given is left holding the loop state at the end, so a
    run can be saved and resumed (docs/gain_of_function_v4/swap_test_phases/resume_from_production_state.py)."""
    from mimo_bridge.body_layout import extend_limits
    from mimo_bridge.lem_model import LemLite
    if state:
        frames, held, prev_exec = state["frames"], state["held"], state["prev_exec"]
        prev_id, prev_gaze, order_seq = state["prev_id"], state["prev_gaze"], int(state["order_seq"])
    else:
        frames = world.frames()
        held = prev_exec = prev_id = prev_gaze = None
        order_seq = 0
    rows = []
    for step in range(start, start + n_steps):
        dev = extend_limits(model.dev_limits(step), len(model.handle_names))
        dev = {**dev, "exploration_noise": float(babble)}
        res = model.step(frames, dev, train=True, step_index=step, executed_action=prev_exec,
                         executed_ctrl_gaze=prev_gaze, executed_order_id=prev_id)
        if step % hold == 0 or held is None:
            held = res.motor_for_action
            order_seq += 1
        frames = world.step(held, res.gaze_out)
        gz = np.asarray(res.gaze_out, np.float32).reshape(len(frames), 2)
        prev_exec = LemLite.gaze_to_handle(held, gz, convention=model.eye_row_convention)
        prev_gaze = gz.copy()
        prev_id = order_seq
        row = dict(step=step, loss=float(res.loss).hex(), grad=float(res.grad_norm).hex(),
                   command=sha(res.motor_for_action), gaze=sha(np.asarray(res.gaze_out, np.float32)),
                   breakdown=hashlib.sha256(json.dumps(sorted((k, float(v).hex()) for k, v in res.breakdown.items())
                                                       ).encode()).hexdigest()[:16],
                   wellness=hashlib.sha256(json.dumps(sorted((k, float(v).hex()) for k, v in res.wellness.items())
                                                      ).encode()).hexdigest()[:16],
                   torch_rng=sha(torch.get_rng_state()),
                   numpy_rng=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()[:16],
                   noise_rng=sha(model.noise_rng.get_state()))
        if digest_every and (step - start + 1) % digest_every == 0:
            row["state"] = state_digest(model)
        rows.append(row)
        if on_step is not None:
            on_step(step, model, res)
    if state is not None:
        state.update(frames=frames, held=held, prev_exec=prev_exec, prev_id=prev_id, prev_gaze=prev_gaze,
                     order_seq=order_seq)
    return rows


def revivals(model):
    """{layer: model step of its last revival} - shows the liveness repair's windows filled during the run."""
    lv = getattr(model, "liveness", None)
    return dict(getattr(lv, "revived_at_step", {}) or {})


DEAD_UNIT_LAYERS = ("prediction_heads.heads.vision.mlp.0", "prediction_heads.heads.tactile.mlp.0",
                    "prediction_heads.heads.vestibular.mlp.0")


@torch.no_grad()
def plant_dead_units(model, layers=DEAD_UNIT_LAYERS, every=2, bias=-3.3, window_rows=120):
    """Make the liveness repair revive units during a short run, so that what its watching windows hold decides
    weights: every `every`-th unit of each layer gets the bias `bias`, and the repair's window is shortened to
    `window_rows` rows (LivenessRepair.revive_min_samples; production's is 800). With -3.3, just under the liveness
    line of -3, most of those units stay under the line through a window and are revived when it fills, and their
    outputs GELU(-3.3 +- a little) are about -0.002, not 0, so the mean output the revival folds into the next
    layer's bias depends on every row the window held - a forward pass that joined the window would change the
    weights (measured: the added passes left watched move the state at model step 9, the first revival). The same
    edits under any code version. Returns the count of planted units."""
    from lem.model.liveness import layer_parts
    n = 0
    for name in layers:
        parts = layer_parts(model, name)
        if parts is None:
            continue
        lin = parts[0]
        lin.bias[::every] = bias
        n += len(range(0, lin.bias.shape[0], every))
    if window_rows and getattr(model, "liveness", None) is not None:
        model.liveness.revive_min_samples = int(window_rows)
    return n
