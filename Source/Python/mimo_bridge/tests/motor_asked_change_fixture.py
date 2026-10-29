# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A small H84-shaped learner driven on synthetic frames: no MuJoCo world, no checkpoint of production, CPU only.

Used by tests/test_motor_asked_change.py and by docs/gain_of_function_v4/motor_encoder_asked_change/planted_defects.py.
The construction follows tests/swap_test_fixture.py of branch fix/swap_test_phases-2026-10-02 (the same launch flags at
a small width, the same synthetic bodies, the trainer's latch rule), written here on its own because that file is not
on this branch's base: train_mimo.parse_args / build_learner build the learner from H84's own launch flags, the real
frame packing and tokenizer make the frames (frame_encoder.encode_frame, datasource._payload_to_frame), and
LemLite.step is driven with a new command latched on model steps that are multiples of --motor-hold and the eyes' row
written from the gaze sent. Replay is on (a 64-model-step ring), so replay_update's motor rows are produced too.

The synthetic bodies are not physics: each of the 26 handles goes a fifth of the way from where it is toward its
commanded position each model step, plus seeded noise; skin contacts, vision patches, audio and vestibular readings are
seeded random numbers. Their numbers are plumbing, not evidence about the body."""
import hashlib
import json
import os

import numpy as np
import torch

# H84's launch flags (Saved/mimo_bridge/training_continuation.json, the command of segment 41s resumed 2026-10-01
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
    a = ["--out", out or os.path.join(os.environ.get("TEMP", "."), "motor_asked_change_fixture_out"),
         "--device", "cpu", "--n-envs", str(n_envs), "--seed", "0"] + list(H84_LEARNER_FLAGS) + list(SMALL)
    a[a.index("--motor-hold") + 1] = str(motor_hold)
    a[a.index("--intention-every") + 1] = str(intention_every)
    return a + list(extra)


def build(args_list, ckpt_dir=None, replay=0):
    """(learner, parsed args) from a launch line, built as the trainer builds it (train_mimo.build_learner).
    replay: the replay ring's size per body (0 = no replay)."""
    from mimo_bridge import train_mimo as T
    args = T.parse_args(args_list)
    args.n_envs = int(args.n_envs)
    model = T.build_learner(args, ckpt_dir)
    model.reset_context(args.n_envs)
    if replay:
        model.enable_replay(args.n_envs, size_per_env=int(replay))
        model.replay_batch = 4
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
        self.n_patches = int(n_patches)
        self.pos = self.rng.normal(0.0, 20.0, (self.B, self.H, 3)).astype(np.float32)
        self.pos[..., 2] += 15.0
        q = self.rng.normal(0.0, 1.0, (self.B, self.H, 4)).astype(np.float32)
        self.quat = q / np.linalg.norm(q, axis=-1, keepdims=True)
        self.vel = np.zeros((self.B, self.H, 3), np.float32)
        self.eye = np.zeros((self.B, 2), np.float32)
        self.frame = 0
        self.t = 0.0

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


def state_digest(model, skip=()):
    """One hash over every parameter and buffer (persistent or not) of every learned component and every Adam moment,
    leaving out the components named in `skip` (their parameters' Adam moments too)."""
    from lem.model.efference import _named_components
    h = hashlib.sha256()
    skipped = set()
    for comp, mod in sorted(_named_components(model).items()):
        if comp in skip:
            skipped.update(id(p) for p in mod.parameters())
            continue
        for n, p in list(mod.named_parameters()) + list(mod.named_buffers()):
            if comp == "liveness_repair" and any(s.replace(".", "__") in n for s in skip):
                skipped.add(id(p))
                continue
            h.update((comp + "." + n).encode())
            h.update(_bytes(p))
    for g in model.optimizer.param_groups:
        for p in g["params"]:
            if id(p) in skipped:
                continue
            for k, v in sorted(model.optimizer.state.get(p, {}).items()):
                if torch.is_tensor(v):
                    h.update(k.encode())
                    h.update(_bytes(v))
    return h.hexdigest()[:16]


def revivals(model):
    """{layer: model step of its last revival}: shows the liveness repair's windows filled during the run."""
    lv = getattr(model, "liveness", None)
    return dict(getattr(lv, "revived_at_step", {}) or {})


DEAD_UNIT_LAYERS = ("prediction_heads.heads.vision.mlp.0", "prediction_heads.heads.tactile.mlp.0",
                    "prediction_heads.heads.vestibular.mlp.0")


@torch.no_grad()
def plant_dead_units(model, layers=DEAD_UNIT_LAYERS, every=2, bias=-3.3, window_rows=120):
    """Make the liveness repair revive units during a short run, so that what its watching windows hold decides
    weights (as tests/swap_test_fixture.py of branch fix/swap_test_phases-2026-10-02 does): every `every`-th unit of
    each layer gets the bias `bias`, just under the liveness line of -3, and the repair's window is shortened to
    `window_rows` rows (production's is 800). A forward pass that joined a watching window would then change the
    weights at the next revival. Returns the count of planted units."""
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


def drive(model, world, n_steps, hold=5, start=0, babble=0.05, digest_every=10, on_step=None, replay_every=0,
          digest_skip=(), state=None, before_step=None):
    """Run n_steps model steps with the trainer's latch rule; one row per model step of what training produced: the
    loss, the gradient norm, the command and gaze sent, the loss breakdown, the wellness values, the three random
    streams, and every digest_every model steps the state digest. replay_every > 0 runs replay_update on those
    model steps (after the live step, as the trainer does). state: the dict a previous call returned in rows[-1]
    ["carry"], to continue a run (frames, held command, executed command). before_step(step, executed command in
    cm or None) is called right before each model step."""
    from mimo_bridge.body_layout import extend_limits
    from mimo_bridge.lem_model import LemLite
    carry = state or {}
    frames = carry.get("frames") or world.frames()
    held, prev_exec, prev_id, prev_gaze = (carry.get(k) for k in ("held", "prev_exec", "prev_id", "prev_gaze"))
    order_seq = int(carry.get("order_seq", 0))
    rows = []
    for step in range(start, start + n_steps):
        dev = extend_limits(model.dev_limits(step), len(model.handle_names))
        dev = {**dev, "exploration_noise": float(babble)}
        if before_step is not None:
            before_step(step, prev_exec)
        res = model.step(frames, dev, train=True, step_index=step, executed_action=prev_exec,
                         executed_ctrl_gaze=prev_gaze, executed_order_id=prev_id)
        replay_loss = None
        if replay_every and step % replay_every == 0 and model.replay is not None:
            rp = model.replay_update(batch=getattr(model, "replay_batch", 4) or 4)
            replay_loss = None if rp is None else float(rp.get("replay_loss", float("nan"))).hex()
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
                   breakdown=hashlib.sha256(json.dumps(sorted((k, float(v).hex()) for k, v in res.breakdown.items()
                                                              if not isinstance(v, dict))
                                                       ).encode()).hexdigest()[:16],
                   wellness=hashlib.sha256(json.dumps(sorted((k, float(v).hex()) for k, v in res.wellness.items())
                                                      ).encode()).hexdigest()[:16],
                   replay=replay_loss, torch_rng=sha(torch.get_rng_state()),
                   numpy_rng=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()[:16],
                   noise_rng=sha(model.noise_rng.get_state()),
                   tokens=sha(torch.cat([model.last_tokens[m].float().reshape(-1) for m in sorted(model.last_tokens)])),
                   predictions=(None if not model._nf_pred else
                                sha(torch.cat([model._nf_pred[m].float().reshape(-1) for m in sorted(model._nf_pred)]))))
        if digest_every and (step - start + 1) % digest_every == 0:
            row["state"] = state_digest(model, skip=digest_skip)
        rows.append(row)
        if on_step is not None:
            on_step(step, model, res)
    if rows:
        rows[-1]["carry"] = dict(frames=frames, held=held, prev_exec=prev_exec, prev_id=prev_id, prev_gaze=prev_gaze,
                                 order_seq=order_seq)
    return rows


class EncoderCalls:
    """Every call of the motor encoder while attached: its call site (the first name of PRIORITY found among the
    calling stack's function names; else "transition", else "step"), a copy of the input rows, the model's step
    count, the caller's tag (self.tag, set by the test: the model step being driven), whether gradients were enabled,
    whether the asked-change path's output weights required a gradient, and whether the path was frozen (its weights
    detached: an imagined row)."""

    # _swap_test_arms: the swap test's added arms (the size-matched command, the shifted-own command and, on the
    # hold's first model step, the absolute roll), built in a method of their own since the swap test's second
    # phase (branch fix/swap_test_phases-2026-10-02); the existing absolute roll at multiples of 20 is still built
    # inside step()
    PRIORITY = ("switch_check", "_swap_test_arms", "replay_update", "_tenure_forward", "_contact_control_forward",
                "_run_imagination_compiled", "generate_action", "_causal_score_candidates", "_causal_plan_record",
                "_plan_chain", "_latent_for_order", "predict_before_execution")

    def __init__(self, model):
        reg = getattr(model.encoder_registry, "_orig_mod", model.encoder_registry)
        self.calls = []
        self.model = model
        self.tag = None
        self._h = reg.encoders["motor"].register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        import sys
        names = []
        f = sys._getframe(1)
        while f is not None and len(names) < 80:
            names.append(f.f_code.co_name)
            f = f.f_back
        site = next((n for n in self.PRIORITY if n in names), None)
        if site is None:
            site = "transition" if "transition" in names else ("step" if "step" in names else "other")
        path = getattr(self.model, "motor_asked_change", None)
        self.calls.append(dict(site=site, stack=names[:30], rows=args[0].detach().float().cpu().clone(),
                               step=int(self.model.step_count), tag=self.tag, grad=bool(torch.is_grad_enabled()),
                               path_trains=(None if path is None else bool(path.up.weight.requires_grad)),
                               frozen=(None if path is None else bool(getattr(path, "frozen_depth", 0)))))
        return None

    def remove(self):
        self._h.remove()
