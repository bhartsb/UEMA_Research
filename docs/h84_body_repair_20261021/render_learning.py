"""Annotate one recorded learning movement for the existing H84 illustrated report.

Owner: the overseer. Retain with this HTML report and its asset group. No physics steps,
training, or checkpoint copies. Run after the arm's saved-state reading; charge
runtime to that reading's existing 900-second allowance. Output names are reused.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/bhart/Documents/UEMA_wt/s4_learning_run_20261022')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--body', required=True, type=int)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--earlier-world')
    args = parser.parse_args()
    started = time.monotonic()
    result = json.loads(Path(args.result).read_text(encoding='utf-8'))
    chosen = next(r for r in result['per_command']
                  if r['body'] == args.body and r['step'] == args.step and r['valid']
                  and r['captured'] and r['command_in_force'] is True)
    # The pinned reader loads the established renderer from MAIN. Its viewer
    # package is also in MAIN, not in the minimal training checkout.
    sys.path.insert(0, str(ROOT / 'Source/Python'))
    reader_path = SOURCE / 'docs/gain_of_function_v4/planner_recovery/s4_end_reading.py'
    spec = importlib.util.spec_from_file_location('h84_saved_learning_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    worlds = [Path(result['run']) / 'observer/world']
    if args.earlier_world:
        worlds.append(Path(args.earlier_world))
    obs = reader.Observer(worlds)
    rs = reader.load_render_saved()
    import numpy as np
    hold = int(result['identity']['motor_hold'])
    steps = {'start': args.step - 1, 'earlier': chosen['goal_step'], 'end': args.step + hold - 1}
    poses = {name: obs.record(args.body, step)[0] for name, step in steps.items()}
    header = obs.headers[args.body]
    wrist_id = int(header['handle_bodies'][header['handle_names'].index('wrist_L')])
    body = rs.R._Body(obs.model_path(args.body), header, ())
    overlays, images = {}, {}
    try:
        body.set_frame(poses['start'])
        body.fit([body.own_geoms(poses[n]) for n in ('start', 'end', 'earlier')], [])
        body.cams = {v: (look, distance * 1.18, az, el) for v, (look, distance, az, el) in body.cams.items()}
        body.place(poses['earlier'])
        earlier = np.array(body.d.xpos[wrist_id])

        def project(point):
            cams = body.renderer.scene.camera
            camera_pos = (np.array(cams[0].pos) + np.array(cams[1].pos)) / 2
            forward, up = np.array(cams[0].forward), np.array(cams[0].up)
            right = np.cross(forward, up)
            vector = body.Rf.T @ (np.asarray(point) - body.o) - camera_pos
            depth = float(vector @ forward)
            near = float(cams[0].frustum_near)
            half_height = (float(cams[0].frustum_top) - float(cams[0].frustum_bottom)) / 2
            nx = float(vector @ right) / depth * near / (half_height * rs.R.BODY_W / rs.R.BODY_H)
            ny = float(vector @ up) / depth * near / half_height
            return [round((nx + 1) * rs.R.BODY_W / 2, 3), round((1 - ny) * rs.R.BODY_H / 2, 3)]

        for view in ('front', 'left', 'right'):
            for name, pose in poses.items():
                filename = body.draw('learning_' + name, view, pose, None, None, out_dir=str(OUT))
                actual = np.array(body.d.xpos[wrist_id])
                images[filename] = sha(OUT / filename)
                overlays[view + '_' + name] = {'actual': project(actual), 'earlier': project(earlier),
                                              'gap_cm': float(np.linalg.norm(actual - earlier) * 100)}
    finally:
        body.close()
    provenance = {'source_result': str(Path(args.result).resolve()), 'source_result_sha256': sha(args.result),
                  'reader_sha256': sha(reader_path), 'generator_sha256': sha(__file__), 'run': result['run'],
                  'body': args.body, 'command_step': args.step, 'record_steps': steps,
                  'model_fingerprint': header['model_fingerprint'], 'physical_result': chosen,
                  'images': images, 'overlays': overlays, 'physics_steps': 0, 'training_updates': 0,
                  'seconds': time.monotonic() - started,
                  'meaning': 'Real saved MIMo poses in the starting chest frame; wrist markers projected through the same camera. Earlier pose is the body at the selected sensor-record time, not a direct planner destination.'}
    (OUT / 'learning_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    (OUT / 'learning_handles.js').write_text('window.H84_LEARNING_HANDLES=' + json.dumps(overlays) + ';\n', encoding='utf-8')
    print(json.dumps({'images': len(images), 'seconds': provenance['seconds'], 'physics_steps': 0}))


if __name__ == '__main__':
    main()
