"""From the saved rows of the two frozen-start runs (same starts, same goals): how far checkpoint 483,411's executed
controller command is from checkpoint 494,025's, per body, against the planners' search region (1.8 cm per coordinate
and 11 degrees per handle around each checkpoint's own controller request, lem.model.efference EFFERENCE_SCALES), and how
far each arm's end pose is from the 494,025 controller's end pose. Read-only."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(r'D:\UEMA_Research')
A = ROOT / 'Saved/mimo_bridge/fixed_probe_set/frozen_483411'
B = ROOT / 'Saved/mimo_bridge/fixed_probe_set/frozen_494025_bodies8'
geometry = json.loads((B / 'hand_geometry.json').read_text(encoding='utf-8'))
names = list(geometry['handle_names'])
body_handles = [i for i, n in enumerate(names) if n != 'gaze']
fingers = sorted({i for h in geometry['hands'] for i in h['finger_indices']})
non_finger = [i for i in body_handles if i not in fingers]
rows = lambda f: {(r['body'], r['goal']): r for r in map(json.loads, (f / 'trials.jsonl').read_text(encoding='utf-8').splitlines())}
a, b = rows(A), rows(B)


def turn_deg(q1, q2):
    return np.degrees((Rotation.from_quat(q1) * Rotation.from_quat(q2).inv()).magnitude())


out = []
for body in range(8):
    key = (body, 'curl_witness')            # the controller arm is the same command for every goal of a body
    ca = np.asarray(a[key]['outcomes']['controller']['canonical_command'], dtype=np.float64)[0]
    cb = np.asarray(b[key]['outcomes']['controller']['canonical_command'], dtype=np.float64)[0]
    pos = np.abs(ca[:, :3] - cb[:, :3])                       # cm per coordinate
    turn = np.array([turn_deg(ca[i, 3:7], cb[i, 3:7]) for i in range(len(names))])
    ea = np.asarray(a[key]['outcomes']['controller']['pose'], dtype=np.float64)[0]
    eb = np.asarray(b[key]['outcomes']['controller']['pose'], dtype=np.float64)[0]
    end_cm = np.linalg.norm((ea[:, :3] - eb[:, :3]) * 200., axis=-1)
    wrist = names.index('wrist_L')
    out.append(dict(body=body,
                    command_largest_coordinate_difference_cm_non_finger=float(pos[non_finger].max()),
                    command_median_coordinate_difference_cm_non_finger=float(np.median(pos[non_finger].max(-1))),
                    command_largest_turn_difference_deg_non_finger=float(turn[non_finger].max()),
                    command_median_turn_difference_deg_non_finger=float(np.median(turn[non_finger])),
                    handles_outside_region=int(sum(1 for i in non_finger if pos[i].max() > 1.8 or turn[i] > 11.0)),
                    non_finger_handles=len(non_finger),
                    left_wrist_command_difference_cm=float(np.linalg.norm(ca[wrist, :3] - cb[wrist, :3])),
                    controller_end_pose_median_handle_distance_cm=float(np.median(end_cm[body_handles])),
                    controller_end_pose_largest_handle_distance_cm=float(end_cm[body_handles].max())))
for o in out:
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items()}))
summary = dict(bodies=8,
               median_over_bodies_largest_command_coordinate_difference_cm=float(np.median([o['command_largest_coordinate_difference_cm_non_finger'] for o in out])),
               median_over_bodies_largest_command_turn_difference_deg=float(np.median([o['command_largest_turn_difference_deg_non_finger'] for o in out])),
               bodies_with_a_handle_outside_the_search_region=int(sum(o['handles_outside_region'] > 0 for o in out)),
               median_handles_outside_region=float(np.median([o['handles_outside_region'] for o in out])),
               median_controller_end_distance_cm=float(np.median([o['controller_end_pose_median_handle_distance_cm'] for o in out])))
print(json.dumps(summary, indent=1))
(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('command_distance.json')).write_text(json.dumps(dict(per_body=out, summary=summary), indent=1))
