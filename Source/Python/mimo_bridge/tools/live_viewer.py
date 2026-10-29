# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A window on the living body, with the camera in your hands.

The training run does not draw this. It publishes the watched body's physical state -
positions, speeds, the mocap bodies, the runtime visibility of every shape - into
<run>/live/state.npz every few model steps (--live-state-every), a few kB per write. This
tool builds its OWN copy of the same scene from the path that state names, puts the body
exactly where the run has it, and hands the picture to MuJoCo's passive viewer, where the
mouse flies the camera freely. Nothing here steps physics, so nothing here can change what
the run learns.

What it costs the run: one small file write per publish, measured as live_state_ms in the
stats. What it costs this machine: one more GL context and whatever the window redraws.

Run:  python -m mimo_bridge.tools.live_viewer                      (newest run)
      python -m mimo_bridge.tools.live_viewer --run <run dir>
      python -m mimo_bridge.tools.live_viewer --run <run dir> --fps 30
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np


def newest_run(root: str) -> str:
    runs = [d for d in glob.glob(os.path.join(root, "Saved", "mimo_bridge", "experiments", "*"))
            if os.path.isdir(d)]
    if not runs:
        raise SystemExit("no run directories under Saved/mimo_bridge/experiments")
    return max(runs, key=os.path.getmtime)


def read_state(path: str):
    """The newest published state, or None while the writer is mid-rename."""
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError):
        return None


def wait_for_state(path: str, seconds: float = 60.0):
    started = time.time()
    while time.time() - started < seconds:
        state = read_state(path)
        if state is not None:
            return state
        time.sleep(0.5)
    raise SystemExit(
        f"no live state at {path} after {seconds:.0f} s. The run must be launched with "
        "--live-state-every N (N model steps between publishes; 5 is about one publish "
        "every four seconds at this run's speed).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    ap.add_argument("--run", default=None, help="run directory (default: the newest)")
    ap.add_argument("--fps", type=float, default=20.0, help="how often the window re-reads the state")
    ap.add_argument("--seconds", type=float, default=0.0, help="close after this long (0 = until closed)")
    a = ap.parse_args(argv)

    run = a.run or newest_run(repo)
    path = os.path.join(run, "live", "state.npz")
    print("run   :", run)
    print("state :", path)
    state = wait_for_state(path)
    scene = str(state.get("scene", np.array("")).item() if hasattr(state.get("scene"), "item")
                else state.get("scene"))
    if not scene or not os.path.isfile(scene):
        raise SystemExit(f"the published state names a scene this machine cannot read: {scene!r}")
    print("scene :", scene)

    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    print(f"window: {model.nq} positions, {model.nbody} bodies, {model.ngeom} shapes. "
          "Drag to orbit, right-drag to pan, scroll to zoom, double-click a body to track it.")

    # ---------------------------------------------------------------- the skin's sensors
    sensors = read_state(os.path.join(run, "live", "sensors.npz"))
    sensor_bodies, sensor_points = [], []
    if sensors is not None:
        ids = np.asarray(sensors["body_ids"]).astype(int)
        counts = np.asarray(sensors["counts"]).astype(int)
        flat = np.asarray(sensors["points"], dtype=np.float64)
        at = 0
        for body_id, n in zip(ids, counts):
            sensor_bodies.append(int(body_id))
            sensor_points.append(flat[at:at + int(n)])
            at += int(n)
        print(f"sensors: {int(counts.sum()):,} touch points over {len(ids)} skin bodies. "
              "Press S to show or hide them.")
    else:
        print("sensors: none published yet (the run writes live/sensors.npz on its first publish)")

    flags_path = os.path.join(run, "live", "viewer_flags.json")
    show = {"sensors": False, "checked": 0.0}

    def on_key(code: int) -> None:
        # S for sensors. The viewer's own keys keep working; this only adds one.
        if code in (ord("s"), ord("S")):
            show["sensors"] = not show["sensors"]
            print("sensors:", "shown" if show["sensors"] else "hidden")

    def poll_flags() -> None:
        """A button on the dashboard writes this file; the window obeys it. Checked once a
        second so a page click reaches the window without polling the disk every frame."""
        now = time.time()
        if now - show["checked"] < 1.0:
            return
        show["checked"] = now
        try:
            import json
            with open(flags_path, encoding="utf-8") as stream:
                want = bool(json.load(stream).get("sensors"))
            if want != show["sensors"]:
                show["sensors"] = want
                print("sensors:", "shown" if want else "hidden", "(from the dashboard)")
        except (OSError, ValueError):
            pass

    def draw_sensors(viewer) -> None:
        """Put one small sphere at every touch point, in the viewer's own user scene, so
        nothing about the body or the run is altered - this is drawing, not physics."""
        scene = viewer.user_scn
        scene.ngeom = 0
        if not show["sensors"]:
            return
        size = np.array([0.0022, 0.0022, 0.0022])
        rgba = np.array([0.15, 0.85, 1.0, 0.85], dtype=np.float32)
        for body_id, local in zip(sensor_bodies, sensor_points):
            if scene.ngeom >= scene.maxgeom:
                break
            pos = data.xpos[body_id]
            mat = data.xmat[body_id].reshape(3, 3)
            world = pos + local @ mat.T
            for point in world:
                if scene.ngeom >= scene.maxgeom:
                    break
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size, point,
                                    np.eye(3).flatten(), rgba)
                scene.ngeom += 1

    def apply(state) -> float:
        """Put this copy of the body exactly where the run has it. mj_forward re-derives
        every dependent quantity; it does NOT integrate, so no physics runs here."""
        for name, target in (("qpos", data.qpos), ("qvel", data.qvel),
                             ("mocap_pos", data.mocap_pos), ("mocap_quat", data.mocap_quat),
                             ("ctrl", data.ctrl)):
            value = state.get(name)
            if value is None:
                continue
            value = np.asarray(value)
            if value.shape == target.shape:
                target[:] = value
        for name, target in (("geom_rgba", model.geom_rgba), ("geom_contype", model.geom_contype)):
            value = state.get(name)
            if value is not None and np.asarray(value).shape == target.shape:
                target[:] = np.asarray(value)
        mujoco.mj_forward(model, data)
        return float(np.asarray(state.get("time", [0.0])).reshape(-1)[0])

    started, last_body_time, stale_since = time.time(), None, None
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            if a.seconds and time.time() - started > a.seconds:
                break
            poll_flags()
            fresh = read_state(path)
            if fresh is not None:
                body_time = apply(fresh)
                draw_sensors(viewer)
                if body_time != last_body_time:
                    last_body_time, stale_since = body_time, None
                elif stale_since is None:
                    stale_since = time.time()
                viewer.sync()
                if stale_since and time.time() - stale_since > 120.0:
                    print(f"the run has published nothing new for "
                          f"{time.time() - stale_since:.0f} s (body clock {body_time:.1f} s)")
                    stale_since = time.time()
            time.sleep(max(0.005, 1.0 / max(1.0, a.fps)))
    print("window closed")


if __name__ == "__main__":
    main()
