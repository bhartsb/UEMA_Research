# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Phase 4, Mode B: MIMo and the model talk ONLY through shared memory, like UE and the model.

    python -m mimo_bridge.run_mode_b --minutes 3 --out Saved/mimo_bridge/phase4

Two processes:
  bridge : MIMo env -> frame bytes -> LEM_Pipeline (+ eyes -> LEM_Vision); reads LEM_Action_Buffer
           (written by the model's own ActionBufferWriter) and moves the body.
  model  : the model's own LEMTokenizer (tokenizer.read_frame, the same reader the UE run uses)
           -> lite LEM -> the model's own ActionBufferWriter. Vision now comes through the
           model tokenizer's CPU path (LEMTokenizer.read_frame reads LEM_Vision when no D3D12
           reader is available; added 2026-08-19), so the model side sees the eyes too.

Pass/fail: both processes run the whole time with no exception, the model processes frames
with increasing frame numbers, the bridge receives actions with increasing frame ids.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mimo_bridge.mode_b")

PIPE_NAME = "LEM_Pipeline"
VIS_NAME = "LEM_Vision"
ACT_NAME = "LEM_Action_Buffer"


# ---------------------------------------------------------------------------- bridge side
def bridge_main(args):
    from tokenizer import config as C
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.datasource import _identity_handles, motor_to_echo
    from mimo_bridge import frame_encoder as FE
    from mimo_bridge.shm_writer import PipelineSHMWriter, VisionSHMWriter, ActionSHMReader
    out = args.out
    summ = {"role": "bridge", "ok": False}
    try:
        pw = PipelineSHMWriter(PIPE_NAME)
        vw = VisionSHMWriter(VIS_NAME) if not args.no_vision_shm else None
        env = BridgeEnv(scene=args.scene, run_dir=os.path.join(out, "scenes"), vision=not args.no_vision_shm,
                        render_every=args.render_every, tag="_modeb")
        raw = env.reset()
        echo = _identity_handles()
        t0 = time.time()
        deadline = t0 + args.minutes * 60
        ar = None
        motor = None; gaze = None
        n_actions = 0; last_fid = 0; n_frames = 0; bad_order = 0
        period = 1.0 / args.hz if args.hz > 0 else 0.0
        while time.time() < deadline:
            t_loop = time.perf_counter()
            # 1. publish the frame the model will read
            buf = FE.encode_frame(frame_number=raw.frame + 1, timestamp=raw.sim_time, dt=raw.dt, wall_clock=time.time() - t0,
                                  gaze_az_deg=raw.gaze_az_el_deg[0], gaze_el_deg=raw.gaze_az_el_deg[1],
                                  bones=raw.proprio.bones, bone_hashes=raw.proprio.hashes, handles=echo,
                                  energy=raw.proprio.energy, body_sets=raw.proprio.body_sets, patches=raw.patches,
                                  collisions=raw.collisions, vision_valid=raw.eye_left is not None)
            pw.write(raw.frame + 1, bytes(buf))
            n_frames += 1
            if vw is not None and raw.eye_left is not None:
                vw.write(raw.frame + 1, raw.eye_left, raw.eye_right)
            # 2. read the newest action, if any
            if ar is None:
                ar = ActionSHMReader(ACT_NAME, retries=1)
                if not ar.is_open:
                    ar = None
            if ar is not None:
                got = ar.read()
                if got is not None:
                    fid, flags, motor, gaze = got
                    if fid <= last_fid:
                        bad_order += 1
                    last_fid = fid; n_actions += 1
            # 3. move the body
            raw = env.step(motor, gaze if gaze is not None else None, None)
            if motor is not None:
                echo = motor_to_echo(motor, None)
            if period > 0:
                dt = period - (time.perf_counter() - t_loop)
                if dt > 0:
                    time.sleep(dt)
        summ.update(ok=True, frames_written=n_frames, actions_received=n_actions, last_action_frame_id=last_fid,
                    out_of_order_actions=bad_order, elapsed_s=time.time() - t0, env_steps=raw.frame)
        env.close(); pw.close()
        if vw is not None:
            vw.close()
    except Exception:
        summ["exception"] = traceback.format_exc()
        log.error("bridge FAILED:\n%s", summ["exception"])
    json.dump(summ, open(os.path.join(out, "bridge_summary.json"), "w"), indent=2)
    return 0 if summ["ok"] else 1


# ---------------------------------------------------------------------------- model side
class _Frame:
    """Minimal BridgeFrame look-alike built from the model tokenizer's TokenizedFrame."""
    def __init__(self, tf):
        self.tokenized = tf
        # vision from the model tokenizer: [N,16,16,3] patches + [N,2] angles (GPU tensors)
        self.vision_patches = tf.vision_patches
        self.vision_angles = tf.vision_angular_pos
        self.frame = tf.metadata.frame_number
        self.sim_time = tf.metadata.timestamp
        self.patches = tf.touch_impulses.numpy() if tf.touch_impulses is not None else np.zeros((256, 10), np.float32)
        self.n_contacts = int(tf.touch_valid_counts[1]) if tf.touch_valid_counts else 0
        self.meta = {}


def model_main(args):
    import torch
    from tokenizer import LEMTokenizer
    from tokenizer.config import TokenizerConfig
    from lem.action.writer import ActionBufferWriter
    from mimo_bridge.lem_model import LemLite
    out = args.out
    summ = {"role": "model", "ok": False}
    try:
        cfg = TokenizerConfig(device=args.device if torch.cuda.is_available() else "cpu",
                              cpu_shared_memory_name=PIPE_NAME, vision_shared_memory_name=VIS_NAME,
                              pipeline_manifest_path=os.path.join(out, "no_manifest.json"),
                              output_manifest_path=os.path.join(out, "tokenizer_manifest.json"))
        tok = LEMTokenizer(cfg)                       # the model's own reader; SHM must exist already
        if not tok.cpu_reader.is_open:
            raise RuntimeError("model side: LEM_Pipeline not found (bridge must start first)")
        model = LemLite(device=args.device, no_vision=args.no_vision_shm, seed=0)
        model.reset_context(1)
        aw = ActionBufferWriter({"action_buffer": {"enabled": True, "shm_name": ACT_NAME, "buffer_size": 1176, "write_timeout_ms": 5}})
        if not aw.enabled:
            raise RuntimeError("model side: could not create LEM_Action_Buffer")
        t0 = time.time(); deadline = t0 + args.minutes * 60
        step = 0; last_fn = 0; bad_order = 0; losses = []; no_frame = 0; frames_with_vision = 0
        while time.time() < deadline:
            tf = tok.read_frame_blocking(timeout_ms=200)
            if tf is None:
                no_frame += 1
                continue
            if tf.metadata.frame_number <= last_fn:
                bad_order += 1
            last_fn = tf.metadata.frame_number
            if tf.vision_patches is not None:
                frames_with_vision += 1
            dev = model.dev_limits(step)
            res = model.step([_Frame(tf)], dev, train=True, step_index=step)
            losses.append(res.loss)
            aw.write(motor_commands=torch.from_numpy(res.motor_for_action[0]),
                     gaze_target=None if res.gaze_out is None else torch.from_numpy(res.gaze_out[0]))
            step += 1
        summ.update(ok=True, model_steps=step, frames_processed=tok.frames_processed, frames_skipped=tok.frames_skipped,
                    last_frame_number=last_fn, out_of_order_frames=bad_order, actions_written=aw.frame_counter,
                    first_loss=losses[0] if losses else None, last_loss=losses[-1] if losses else None,
                    nan_losses=int(sum(1 for l in losses if not np.isfinite(l))), elapsed_s=time.time() - t0,
                    empty_reads=no_frame, frames_with_vision=frames_with_vision)
        aw.close(); tok.close()
    except Exception:
        summ["exception"] = traceback.format_exc()
        log.error("model FAILED:\n%s", summ["exception"])
    json.dump(summ, open(os.path.join(out, "model_summary.json"), "w"), indent=2)
    return 0 if summ["ok"] else 1


# ---------------------------------------------------------------------------- orchestrator
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["orchestrate", "bridge", "model"], default="orchestrate")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join("Saved", "mimo_bridge", "phase4"))
    ap.add_argument("--scene", default="reach")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hz", type=float, default=30.0, help="bridge frame rate cap (UE runs ~40 to 50)")
    ap.add_argument("--render-every", type=int, default=2)
    ap.add_argument("--no-vision-shm", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.role == "bridge":
        sys.exit(bridge_main(args))
    if args.role == "model":
        sys.exit(model_main(args))
    py = sys.executable
    common = ["--minutes", str(args.minutes), "--out", args.out, "--scene", args.scene, "--device", args.device,
              "--hz", str(args.hz), "--render-every", str(args.render_every)] + (["--no-vision-shm"] if args.no_vision_shm else [])
    lb = open(os.path.join(args.out, "bridge.log"), "w"); lm = open(os.path.join(args.out, "model.log"), "w")
    pb = subprocess.Popen([py, "-m", "mimo_bridge.run_mode_b", "--role", "bridge"] + common, cwd=ROOT, stdout=lb, stderr=subprocess.STDOUT)
    time.sleep(12)   # let the bridge create the shared memory and the env first
    pm = subprocess.Popen([py, "-m", "mimo_bridge.run_mode_b", "--role", "model"] + common, cwd=ROOT, stdout=lm, stderr=subprocess.STDOUT)
    rb = pb.wait(); rm = pm.wait()
    bs = json.load(open(os.path.join(args.out, "bridge_summary.json"))) if os.path.exists(os.path.join(args.out, "bridge_summary.json")) else {}
    ms = json.load(open(os.path.join(args.out, "model_summary.json"))) if os.path.exists(os.path.join(args.out, "model_summary.json")) else {}
    ok = (rb == 0 and rm == 0 and bs.get("actions_received", 0) > 0 and ms.get("frames_processed", 0) > 0
          and ms.get("out_of_order_frames", 1) == 0 and bs.get("out_of_order_actions", 1) == 0)
    summ = {"ok": ok, "bridge": bs, "model": ms, "bridge_rc": rb, "model_rc": rm}
    json.dump(summ, open(os.path.join(args.out, "mode_b_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
