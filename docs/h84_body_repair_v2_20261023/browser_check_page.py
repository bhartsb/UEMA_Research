"""Real-browser check of one page: python browser_check_page.py <page.html> <out-prefix> <section-id> [<section-id> ...]
Screenshots at desktop (1440) and phone (390) width per section, broken anchors, runtime errors, sideways overflow,
and a receipt <out-prefix>.json beside this script. No physics or training."""
from pathlib import Path
import base64, json, subprocess, sys, tempfile, time, urllib.request, shutil, hashlib
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
PAGE = Path(sys.argv[1]); PAGE = PAGE if PAGE.is_absolute() else ROOT / PAGE
PREFIX = sys.argv[2]; SECTIONS = sys.argv[3:]
helper = ROOT / 'Saved/gof_coordination/audit_20261017/browser_audit.py'
ns = {'__file__': str(helper)}; exec(compile(helper.read_text().split("edge='")[0], str(helper), 'exec'), ns)
WS = ns['WS']; edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
profile = Path(tempfile.mkdtemp(prefix='h84_page_check_')); assert profile.resolve().parent == Path(tempfile.gettempdir()).resolve()
proc = subprocess.Popen([edge, '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=9246', '--user-data-dir=' + str(profile), 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
ws = None; checks = {'page': str(PAGE)}
try:
    for _ in range(100):
        try: targets = json.load(urllib.request.urlopen('http://127.0.0.1:9246/json/list', timeout=1)); break
        except OSError: time.sleep(.1)
    else: raise RuntimeError('browser did not open')
    ws = WS(next(t['webSocketDebuggerUrl'] for t in targets if t['type'] == 'page')); ws.call('Page.enable'); ws.call('Runtime.enable')
    ws.call('Page.navigate', {'url': PAGE.as_uri()})
    for _ in range(100):
        if ws.js("document.readyState==='complete'"): break
        time.sleep(.05)
    time.sleep(.3)
    checks['viewports'] = []; checks['screenshots'] = []
    for name, w, h in [('desktop', 1440, 1150), ('narrow', 390, 1000)]:
        ws.call('Emulation.setDeviceMetricsOverride', {'width': w, 'height': h, 'deviceScaleFactor': 1, 'mobile': name == 'narrow'}); time.sleep(.25)
        checks['viewports'].append({'name': name, **ws.js('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})')})
        for section in SECTIONS:
            ok = ws.js("!!document.getElementById('" + section + "')")
            if not ok: checks.setdefault('missing_sections', []).append(section); continue
            ws.js("document.getElementById('" + section + "').scrollIntoView()"); time.sleep(.06)
            shot = ws.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
            (OUT / ('review_' + PREFIX + '_' + name + '_' + section + '.png')).write_bytes(base64.b64decode(shot['data'])); checks['screenshots'].append('review_' + PREFIX + '_' + name + '_' + section + '.png')
    checks['broken_anchors'] = ws.js("[...document.querySelectorAll('a[href^=\"#\"]')].map(a=>a.hash.slice(1)).filter(id=>!document.getElementById(id))")
    checks['table_rows'] = ws.js("[...document.querySelectorAll('tbody')].map(t=>t.querySelectorAll('tr').length)")
    checks['errors'] = [e for e in ws.events if e.get('method') == 'Runtime.exceptionThrown']
    checks['no_overflow'] = all(v['scrollWidth'] <= v['width'] for v in checks['viewports'])
    checks['html_sha256'] = hashlib.sha256(PAGE.read_bytes()).hexdigest(); checks['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    pj = ROOT / 'docs/h84_reach_plan_visual_20261023/plan_pictures.json'
    if PREFIX == 'visual' and pj.exists():
        pr = json.loads(pj.read_text(encoding='utf-8')); checks['pictures_generated'] = pr.get('started'); checks['pictures_count'] = len(pr.get('images', {}))
finally:
    if ws:
        try: ws.send({'id': 999999, 'method': 'Browser.close'}); ws.s.close()
        except OSError: pass
    try: proc.wait(timeout=8)
    except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=4)
    for _ in range(20):
        try:
            assert profile.resolve().parent == Path(tempfile.gettempdir()).resolve() and profile.name.startswith('h84_page_check_')
            shutil.rmtree(profile); break
        except PermissionError: time.sleep(.2)
    checks['temporary_browser_profile_removed'] = not profile.exists(); (OUT / ('browser_checks_' + PREFIX + '.json')).write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks))
