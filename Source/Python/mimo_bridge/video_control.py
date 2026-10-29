# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Observer-only descriptions of the command actually sent to the body.

These labels are rendered into saved footage, never into sensory input. They
describe control ownership, not intention or whether the body physically stopped.
"""
import numpy as np

VERSION = 'body_control_video_v1'
LABELS = {
    'scheduled_hold': ('SCHEDULED STILLNESS', 'Trainer holds body; eyes can still move'),
    'model': ('MODEL BODY COMMAND', 'May include exploration; purpose untested'),
    'random': ('RANDOM BODY COMMAND', 'Random-action comparison; not a learned choice'),
    'unrecorded': ('CONTROL SOURCE NOT RECORDED', 'Movement alone does not establish purpose'),
}


def body_control(scheduled_hold, random_actions=False):
    if scheduled_hold:
        return 'scheduled_hold'
    if random_actions:
        return 'random'
    return 'model'


def caregiver_caption(env):
    c=getattr(env,'caregiver',None)
    if c is None:return None
    phases={'shared':'Nearby','quiet':'Quiet together','farewell':'Saying goodbye',
            'leaving':'Leaving','away':'Away','returning':'Returning'}
    s=c.state;phase=s.get('adult_phase','shared')
    speech=env._audio.speech
    # The just-emitted samples can include the last part of an utterance even
    # after busy becomes false. Describe this captured audio interval.
    speaking=speech.busy or bool(len(speech.last_pcm) and np.max(np.abs(speech.last_pcm))>1e-5)
    text=c.bank.texts.get(speech.key,'') if speaking else ''
    return dict(phase=phase,label=phases.get(phase,phase),text=text)


def annotate(frame, control, caregiver=None):
    """Add a readable observer banner to a copy; keep the sensor image intact."""
    from PIL import Image, ImageDraw, ImageFont
    mode = control if control in LABELS else 'unrecorded'
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8)).copy()
    draw = ImageDraw.Draw(im)
    width, height = im.size
    size = max(10, min(17, width // 28))
    font = ImageFont.load_default(size=size)
    title, detail = LABELS[mode]
    # Fit the longer second line on the actual camera resolution.
    while draw.textbbox((0, 0), detail, font=font)[2] > width - 12 and size > 8:
        size -= 1
        font = ImageFont.load_default(size=size)
    band = min(height, 2 * size + 18)
    draw.rectangle((0, 0, width, band), fill=(16, 23, 29))
    accent = (255, 211, 92) if mode == 'scheduled_hold' else (228, 237, 242)
    draw.text((6, 4), title, font=font, fill=accent)
    draw.text((6, size + 9), detail, font=font, fill=(228, 237, 242))
    if caregiver:
        lines=['Caregiver: '+caregiver['label']]
        words=caregiver.get('text','').split();line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if line and draw.textbbox((0,0),candidate,font=font)[2]>width-12:
                lines.append(line);line=word
            else:line=candidate
        if line:lines.append(line)
        bottom=min(height-band,len(lines)*(size+4)+10)
        draw.rectangle((0,height-bottom,width,height),fill=(16,23,29))
        for i,line in enumerate(lines):draw.text((6,height-bottom+4+i*(size+4)),line,font=font,fill=(228,237,242))
    return np.asarray(im).copy()
