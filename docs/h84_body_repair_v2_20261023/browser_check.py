"""Real-browser check of H84_BODY_REPAIR_EXPLAINED_v2_2026-10-23.html; existing asset groups; no physics or training."""
from pathlib import Path
import base64, json, subprocess, tempfile, time, urllib.request, shutil, hashlib
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
PAGE = ROOT / 'docs/H84_BODY_REPAIR_EXPLAINED_v2_2026-10-23.html'
helper = ROOT / 'Saved/gof_coordination/audit_20261017/browser_audit.py'
ns = {'__file__': str(helper)}; exec(compile(helper.read_text().split("edge='")[0], str(helper), 'exec'), ns)
WS = ns['WS']; edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
profile = Path(tempfile.mkdtemp(prefix='h84_body_report_v2_')); assert profile.resolve().parent == Path(tempfile.gettempdir()).resolve()
proc = subprocess.Popen([edge, '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=9244', '--user-data-dir=' + str(profile), 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
ws = None; checks = {}
try:
    for _ in range(100):
        try: targets = json.load(urllib.request.urlopen('http://127.0.0.1:9244/json/list', timeout=1)); break
        except OSError: time.sleep(.1)
    else: raise RuntimeError('browser did not open')
    ws = WS(next(t['webSocketDebuggerUrl'] for t in targets if t['type'] == 'page')); ws.call('Page.enable'); ws.call('Runtime.enable')
    ws.call('Page.navigate', {'url': PAGE.as_uri()})
    def loaded():
        for _ in range(100):
            if ws.js("document.readyState==='complete' && [...document.images].every(i=>i.complete&&i.naturalWidth>0)"): return
            time.sleep(.05)
        raise RuntimeError('images failed to load')
    loaded(); checks['viewports'] = []
    sections = ['top', 'picture', 'parts', 'goal', 'command', 'motion', 'frames', 'calc-stages', 'faults', 'score', 'records', 'lesson', 'questions', 'sources']
    for name, w, h in [('desktop', 1440, 1150), ('narrow', 390, 1000)]:
        ws.call('Emulation.setDeviceMetricsOverride', {'width': w, 'height': h, 'deviceScaleFactor': 1, 'mobile': name == 'narrow'})
        time.sleep(.2)
        checks['viewports'].append({'name': name, **ws.js('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})')})
        for section in sections:
            ws.js("document.getElementById('" + section + "').scrollIntoView()"); time.sleep(.06)
            shot = ws.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
            (OUT / ('review_' + name + '_' + section + '.png')).write_bytes(base64.b64decode(shot['data']))
    ws.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1150, 'deviceScaleFactor': 1, 'mobile': False})
    checks['frames'] = []
    for view in ['front', 'left', 'right']:
        ws.js("document.querySelector('[data-pview=" + view + "]').click()"); loaded()
        for i in range(6):
            ws.js("document.querySelector('[data-frame=\"" + str(i) + "\"]').click()"); loaded()
            checks['frames'].append(ws.js("({view:'" + view + "',frame:" + str(i) + ",src:document.getElementById('frame-img').src.split('/').pop(),loaded:document.getElementById('frame-img').naturalWidth>0,title:document.getElementById('frame-title').textContent,marks:document.getElementById('frame-svg').querySelectorAll('circle,rect,line').length})"))
    ws.js("document.querySelector('[data-pview=front]').click();document.querySelector('[data-frame=\"5\"]').click();document.getElementById('frames').scrollIntoView()"); loaded()
    shot = ws.call('Page.captureScreenshot', {'format': 'png'}); (OUT / 'review_desktop_frames_250ms.png').write_bytes(base64.b64decode(shot['data']))
    checks['play'] = ws.js("(()=>{const p=document.getElementById('play');p.click();const on=p.textContent;p.click();return {started:on,stopped:p.textContent}})()")
    checks['a2_gallery'] = []
    for case in ['largest_disagreement', 'typical_improving', 'typical_worsening']:
        for stage in range(5):
            ws.js("document.getElementById('a2-case').value='" + case + "';document.getElementById('a2-case').dispatchEvent(new Event('change'));document.querySelector('[data-a2=\"" + str(stage) + "\"]').click()")
            loaded(); checks['a2_gallery'].append(ws.js("({case:'" + case + "',stage:" + str(stage) + ",loaded:document.getElementById('a2-image').naturalWidth>0,caption:document.getElementById('a2-caption').textContent.slice(0,60)})"))
    checks['calc_stages'] = []
    ws.call('Emulation.setDeviceMetricsOverride', {'width': 390, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': True})
    for i in range(3):
        ws.js("document.querySelector('[data-group=calc-stages][data-stage=\"" + str(i) + "\"]').click()")
        checks['calc_stages'].append(ws.js("({stage:" + str(i) + ",visible:[...document.getElementById('calc-stages').children].filter(x=>getComputedStyle(x).display!=='none').map(x=>x.querySelector('h3').textContent)})"))
    ws.js("document.querySelector('[data-group=calc-stages][data-stage=\"0\"]').click()")
    ws.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1150, 'deviceScaleFactor': 1, 'mobile': False})
    checks['tables'] = ws.js("({command_rows:document.querySelectorAll('#command-table tr').length,path_rows:document.querySelectorAll('#path-table tr').length,angle_rows:document.querySelectorAll('#angles-table tr').length})")
    checks['broken_anchors'] = ws.js("[...document.querySelectorAll('a[href^=\"#\"]')].map(a=>a.hash.slice(1)).filter(id=>!document.getElementById(id))")
    checks['errors'] = [e for e in ws.events if e.get('method') == 'Runtime.exceptionThrown']; checks['images'] = ws.js('document.images.length')
    checks['no_overflow'] = all(v['scrollWidth'] <= v['width'] for v in checks['viewports']); checks['question'] = 'What can improve clarity?'
    checks['html_sha256'] = hashlib.sha256(PAGE.read_bytes()).hexdigest(); checks['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
finally:
    if ws:
        try: ws.send({'id': 999999, 'method': 'Browser.close'}); ws.s.close()
        except OSError: pass
    try: proc.wait(timeout=8)
    except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=4)
    for _ in range(20):
        try:
            assert profile.resolve().parent == Path(tempfile.gettempdir()).resolve() and profile.name.startswith('h84_body_report_v2_')
            shutil.rmtree(profile); break
        except PermissionError: time.sleep(.2)
    checks['temporary_browser_profile_removed'] = not profile.exists(); (OUT / 'browser_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks))
