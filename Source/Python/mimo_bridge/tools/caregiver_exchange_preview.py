# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Record a complete adult departure and return, with a declared shortened initial wait."""
import argparse
import json
from pathlib import Path
from mimo_bridge.env import BridgeEnv
from mimo_bridge.caregiver_interaction import initialize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--speech',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    e=BridgeEnv(scene='crib',run_dir=str(args.out/'scene'),age=24,vision=True,eye_size=128,
        audio_real=True,mobile=True,critters=True,caregiver_mode='responsive',
        caregiver_speech_dir=str(args.speech.resolve()))
    phases=[];previous=None
    try:
        e.reset();initialize(e.caregiver,e);e.caregiver.state['departure_at']=6.
        steps=int(125/e.dt);e.start_video(str(args.out/'exchange.mp4'),steps,size=480,every=10)
        for i in range(steps):
            e.step(render=i%10==0)
            phase=e.caregiver.state['adult_phase']
            if phase!=previous:
                phases.append(dict(time=e.caregiver.state['time'],phase=phase,
                    adult_visibility=e.caregiver.state['adult_visibility']))
                previous=phase
                if phase in ('shared','leaving','away','returning'):
                    e.save_views(str(args.out/(f'{len(phases)}_{phase}.png')),480)
        events=[dict(ev,text=e.caregiver.bank.texts[ev['phrase']]) for ev in e.caregiver.state['events'] if 'phrase' in ev]
        (args.out/'review.json').write_text(json.dumps(dict(phases=phases,utterances=events,
            demonstration='Passive body. Initial departure wait shortened to six simulated seconds for viewing; all movement, speech and absence durations use the production controller.'),indent=2))
    finally:e.close()
    html='''<!doctype html><meta charset="utf-8"><title>Caregiver exchanges</title>
<style>body{font:18px/1.6 system-ui;background:#f3f0e9;color:#273e44;max-width:1000px;margin:40px auto;padding:20px}video,img{width:100%;border-radius:12px}button{font:inherit;padding:8px 14px;margin:5px;border:1px solid #829b98;border-radius:8px;background:white}p{max-width:850px}</style>
<h1>Speaking, leaving, and coming back</h1><p>This is an environment demonstration with a passive body. Only the first waiting period was shortened for this recording. It does not demonstrate learned behavior.</p>
<video id="v" src="exchange.mp4" controls playsinline></video><div id="chapters"></div><p>The captions identify the adult's state and words. The baby's sensory images contain neither captions nor written words.</p>
<script>const phases=PHASES;for(const p of phases){const b=document.createElement('button');b.textContent=p.phase.replaceAll('_',' ');b.onclick=()=>{v.currentTime=p.time;v.play()};chapters.append(b)}</script>'''
    (args.out/'index.html').write_text(html.replace('PHASES',json.dumps(phases)),encoding='utf-8')


if __name__=='__main__':main()
