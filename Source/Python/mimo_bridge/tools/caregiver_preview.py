# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Record the actual caregiver world and ear audio, without claiming learned behavior."""
from pathlib import Path
import argparse
import json
from mimo_bridge.env import BridgeEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--speech', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=50.)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = BridgeEnv(scene='crib', run_dir=str(args.out / 'scene'), age=24,
                    vision=True, eye_size=128, audio_real=True, mobile=True, critters=True,
                    caregiver_mode='responsive', caregiver_speech_dir=str(args.speech.resolve()),
                    touch_geometry='surface')
    try:
        env.reset()
        count = int(args.seconds / env.dt)
        env.start_video(str(args.out / 'caregiver.mp4'), count, size=480, every=10)
        for step in range(count):
            env.step(render=step % 10 == 0)
            if step in (20, 2500, 4900):
                env.save_views(str(args.out / f'world_{step}.png'), size=384)
        env.caregiver.export_events(args.out / 'events.json')
        events = [dict(e,text=env.caregiver.bank.texts[e['phrase']],
                       duration=len(env.caregiver.bank.clips[e['phrase']])/env._audio.speech.sample_rate)
                  for e in env.caregiver.state['events'] if 'phrase' in e]
        (args.out / 'transcript.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    finally:
        env.close()
    page = '''<!doctype html><meta charset="utf-8"><title>Caregiver world — implementation preview</title>
<style>body{font:18px/1.6 system-ui;margin:40px auto;max-width:1100px;padding:0 20px;background:#101820;color:#edf3f5}
video{width:100%;border-radius:14px}h1{font-size:32px}p{max-width:800px}.status{padding:18px;background:#243746;border-radius:10px}
a{color:#8edee7}img{width:100%}</style>
<h1>A world that changes and responds</h1>
<p>This is an environment check with a passive body. It does not demonstrate learned intention.
Its speech reaches the infant's audio input; the recording uses the same ear waveforms.</p>
<video id="video" src="caregiver.mp4" controls preload="metadata"></video>
<p class="status" id="status">Play to follow the caregiver's actions.</p>
<p>The caregiver presents a ball, a soft contact pad and a ring, pauses, and carries objects away.
It can answer sustained looking or infant contact. No points are awarded for looking at a toy.</p>
<p><a href="events.json">Recorded adult movements and speech events</a> · <a href="transcript.json">Speech timing</a></p>
<img src="world_20.png" alt="Scene from five directions and the infant's left eye">
<script>const events=EVENTS;const labels={ball:'Here is the ball.',cloth:'Here is the soft cloth.',ring:'Here is the ring.',
pause:'The caregiver pauses.',away:'The caregiver carries the object away.',look:'The caregiver answers sustained looking.',touch:'The caregiver answers contact.'};
video.ontimeupdate=()=>{let e=events.filter(e=>e.time<=video.currentTime&&video.currentTime<e.time+e.duration).at(-1);status.textContent=e?e.text:'A pause in the conversation.';};</script>'''
    # Explicit lookup avoids the browser's window.status string property.
    page = page.replace('status.textContent=', "document.getElementById('status').textContent=")
    (args.out / 'index.html').write_text(page.replace('EVENTS', json.dumps(events)), encoding='utf-8')


if __name__ == '__main__':
    main()
