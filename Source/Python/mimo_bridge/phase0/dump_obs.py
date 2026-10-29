# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Phase 0: count touch sensor points per body and dump one observation of every modality."""
import os, json, numpy as np, gymnasium as gym
import mimo_infant.simulation
import mujoco
env = gym.make("MIMoBenchV2-v0").unwrapped
obs, _ = env.reset()
m, d = env.model, env.data
out = {}
# --- touch points per body (default v2 scales) ---
per_body = {}
for bid in env.touch.sensing_bodies():
    per_body[m.body(bid).name] = {"body_id": int(bid), "n_points": int(env.touch.get_sensor_count(bid)),
                                  "scale_m": float(env.touch.sensor_scales[bid])}
out["touch_points_per_body"] = per_body
out["touch_points_total"] = int(sum(v["n_points"] for v in per_body.values()))
out["touch_sensing_bodies"] = len(per_body)
out["touch_output_shape"] = list(env.touch.get_touch_obs().shape)
out["touch_function"] = env.touch.touch_type; out["response_function"] = env.touch.response_type
# --- model counts ---
out["nbody"] = int(m.nbody); out["njnt"] = int(m.njnt); out["nu"] = int(m.nu)
out["n_mimo_joints(robot:*)"] = int(len(env.mimo_joints)); out["n_mimo_actuators(act:*)"] = int(len(env.mimo_actuators))
out["timestep"] = float(m.opt.timestep); out["frame_skip"] = int(env.frame_skip); out["env_dt"] = float(env.dt)
out["body_names"] = [m.body(i).name for i in range(m.nbody)]
out["actuator_names"] = [m.actuator(i).name for i in range(m.nu)]
out["joint_names"] = [m.joint(i).name for i in range(m.njnt)]
out["camera_names"] = [m.camera(i).name for i in range(m.ncam)]
out["sensor_names"] = [m.sensor(i).name for i in range(m.nsensor)]
# --- one observation of every modality ---
for _ in range(5):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
def summ(a):
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "nonzero": int(np.count_nonzero(a))}
out["observation_summary"] = {k: summ(v) for k, v in obs.items()}
out["proprio_components"] = {k: [str(x) for x in v][:5] + (["..."] if len(v) > 5 else []) for k, v in env.proprioception.sensor_names.items()}
out["proprio_component_sizes"] = {k: len(v) for k, v in env.proprioception.sensor_names.items()}
out["action_space"] = {"shape": list(env.action_space.shape), "low": float(env.action_space.low.min()), "high": float(env.action_space.high.max())}
out["ncon_after_5_steps"] = int(d.ncon)
np.savez_compressed("one_observation.npz", **{k: np.asarray(v) for k, v in obs.items()})
json.dump(out, open("phase0_counts.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k not in ("body_names","actuator_names","joint_names","sensor_names","touch_points_per_body")}, indent=2))
print("touch points per body:"); 
for k,v in sorted(per_body.items(), key=lambda kv: -kv[1]["n_points"]): print(f"  {k:22s} {v['n_points']:5d}  scale={v['scale_m']}")
