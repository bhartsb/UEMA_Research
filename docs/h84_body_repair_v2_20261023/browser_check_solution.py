"""Real-browser check of H84_REACH_SOLUTION_2026-10-23.html; no physics or training."""
from pathlib import Path
import base64, json, subprocess, tempfile, time, urllib.request, shutil, hashlib
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
PAGE = ROOT / 'docs/H84_REACH_SOLUTION_2026-10-23.html'
helper = ROOT / 'Saved/gof_coordination/audit_20261017/browser_audit.py'
ns = {'__file__': str(helper)}; exec(compile(helper.read_text().split("edge='")[0], str(helper), 'exec'), ns)
WS = ns['WS']; edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
profile = Path(tempfile.mkdtemp(prefix='h84_solution_')); assert profile.resolve().parent == Path(tempfile.gettempdir()).resolve()
proc = subprocess.Popen([edge, '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=9245', '--user-data-dir=' + str(profile), 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
ws = None; checks = {}
try:
    for _ in range(100):
        try: targets = json.load(urllib.request.urlopen('http://127.0.0.1:9245/json/list', timeout=1)); break
        except OSError: time.sleep(.1)
    else: raise RuntimeError('browser did not open')
    ws = WS(next(t['webSocketDebuggerUrl'] for t in targets if t['type'] == 'page')); ws.call('Page.enable'); ws.call('Runtime.enable')
    ws.call('Page.navigate', {'url': PAGE.as_uri()})
    for _ in range(100):
        if ws.js("document.readyState==='complete'"): break
        time.sleep(.05)
    time.sleep(.3)
    checks['viewports'] = []
    for name, w, h in [('desktop', 1440, 1150), ('narrow', 390, 1000)]:
        ws.call('Emulation.setDeviceMetricsOverride', {'width': w, 'height': h, 'deviceScaleFactor': 1, 'mobile': name == 'narrow'}); time.sleep(.2)
        checks['viewports'].append({'name': name, **ws.js('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})')})
        for section in ['top', 'today', 'why', 'steps', 'decisions', 'sources']:
            ws.js("document.getElementById('" + section + "').scrollIntoView()"); time.sleep(.06)
            shot = ws.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
            (OUT / ('review_solution_' + name + '_' + section + '.png')).write_bytes(base64.b64decode(shot['data']))
    ws.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1150, 'deviceScaleFactor': 1, 'mobile': False})
    checks['tables'] = ws.js("({summary:document.querySelectorAll('#summary-table tr').length,held:document.querySelectorAll('#held-table tr').length,all:document.querySelectorAll('#all-table tr').length,traj:document.querySelectorAll('#traj-table tr').length})")
    checks['broken_anchors'] = ws.js("[...document.querySelectorAll('a[href^=\"#\"]')].map(a=>a.hash.slice(1)).filter(id=>!document.getElementById(id))")
    checks['errors'] = [e for e in ws.events if e.get('method') == 'Runtime.exceptionThrown']
    checks['no_overflow'] = all(v['scrollWidth'] <= v['width'] for v in checks['viewports'])
    checks['html_sha256'] = hashlib.sha256(PAGE.read_bytes()).hexdigest(); checks['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
finally:
    if ws:
        try: ws.send({'id': 999999, 'method': 'Browser.close'}); ws.s.close()
        except OSError: pass
    try: proc.wait(timeout=8)
    except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=4)
    for _ in range(20):
        try:
            assert profile.resolve().parent == Path(tempfile.gettempdir()).resolve() and profile.name.startswith('h84_solution_')
            shutil.rmtree(profile); break
        except PermissionError: time.sleep(.2)
    checks['temporary_browser_profile_removed'] = not profile.exists(); (OUT / 'browser_checks_solution.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks))
