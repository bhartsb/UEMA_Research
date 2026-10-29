"""Real-browser report review; existing asset group; no physics or training."""
from pathlib import Path
import base64,json,subprocess,tempfile,time,urllib.request,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
helper=ROOT/'Saved/gof_coordination/audit_20261017/browser_audit.py'
ns={'__file__':str(helper)};exec(compile(helper.read_text().split("edge='")[0],str(helper),'exec'),ns)
WS=ns['WS'];edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
profile=Path(tempfile.mkdtemp(prefix='h84_body_report_'));assert profile.resolve().parent==Path(tempfile.gettempdir()).resolve()
proc=subprocess.Popen([edge,'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=9243','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
ws=None;checks={}
try:
 for _ in range(100):
  try:targets=json.load(urllib.request.urlopen('http://127.0.0.1:9243/json/list',timeout=1));break
  except OSError:time.sleep(.1)
 else:raise RuntimeError('browser did not open')
 ws=WS(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'));ws.call('Page.enable');ws.call('Runtime.enable')
 ws.call('Page.navigate',{'url':(ROOT/'docs/H84_BODY_REPAIR_EXPLAINED_2026-10-21.html').as_uri()})
 def loaded():
  for _ in range(100):
   if ws.js("document.readyState==='complete' && [...document.images].every(i=>i.complete&&i.naturalWidth>0)"):return
   time.sleep(.05)
  raise RuntimeError('images failed to load')
 loaded();checks['viewports']=[]
 sections=['a2-diagnosis','consistency-lesson','top','original','joints','joint-calculation','execution','rule','gaps','evidence','plan','field-map','geometry','checklist','reviews']
 for name,w,h in [('desktop',1440,1150),('narrow',390,1000)]:
  ws.call('Emulation.setDeviceMetricsOverride',{'width':w,'height':h,'deviceScaleFactor':1,'mobile':name=='narrow'})
  checks['viewports'].append({'name':name,**ws.js('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})')})
  for section in sections:
   ws.js("document.getElementById('"+section+"').scrollIntoView()");time.sleep(.06)
   shot=ws.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
   (OUT/('review_'+name+'_'+section+'.png')).write_bytes(base64.b64decode(shot['data']))
 checks['a2_gallery']=[]
 for case in ['largest_disagreement','typical_improving','typical_worsening']:
  for stage in range(5):
   ws.js("document.getElementById('a2-case').value='"+case+"';document.getElementById('a2-case').dispatchEvent(new Event('change'));document.querySelector('[data-a2=\""+str(stage)+"\"]').click()")
   loaded();checks['a2_gallery'].append(ws.js("({case:'"+case+"',stage:"+str(stage)+",loaded:document.getElementById('a2-image').naturalWidth>0,caption:document.getElementById('a2-caption').textContent})"))
 ws.js("document.getElementById('a2-case').value='largest_disagreement';document.getElementById('a2-case').dispatchEvent(new Event('change'));document.querySelector('[data-a2=\"3\"]').click()")
 loaded()
 for name,w,h in [('desktop',1440,1150),('narrow',390,1000)]:
  ws.call('Emulation.setDeviceMetricsOverride',{'width':w,'height':h,'deviceScaleFactor':1,'mobile':name=='narrow'})
  ws.js("document.getElementById('a2-caption').scrollIntoView()")
  shot=ws.call('Page.captureScreenshot',{'format':'png'});(OUT/('review_'+name+'_a2-selected.png')).write_bytes(base64.b64decode(shot['data']))
 checks['phone_stage_controls']=[]
 for group in ['joint-stages','calc-stages']:
  for i in range(3):
   ws.js("document.querySelector('[data-group="+group+"][data-stage=\""+str(i)+"\"]').click()")
   checks['phone_stage_controls'].append(ws.js("({group:'"+group+"',stage:"+str(i)+",visible:[...document.getElementById('"+group+"').children].filter(x=>getComputedStyle(x).display!=='none').map(x=>x.querySelector('h3').textContent)})"))
  ws.js("document.getElementById('"+group+"').scrollIntoView()")
  shot=ws.call('Page.captureScreenshot',{'format':'png'});(OUT/('review_narrow_'+group+'_selected.png')).write_bytes(base64.b64decode(shot['data']))
  ws.js("document.querySelector('[data-group="+group+"][data-stage=\"0\"]').click()")
 checks['camera_controls']=[]
 for view in ['left','right','front']:
  ws.js("document.querySelector('[data-view="+view+"]').click()");loaded()
  checks['camera_controls'].append({'view':view,'match':ws.js("[...document.querySelectorAll('img[data-state],img[data-joint-state]')].every(i=>i.src.includes('body_"+view+"_'))")})
 checks['joint_controls']=[]
 for i in range(7):
  ws.js("document.getElementById('joint-choice').value='"+str(i)+"';document.getElementById('joint-choice').dispatchEvent(new Event('change'))")
  checks['joint_controls'].append(ws.js("({name:document.getElementById('joint-choice').selectedOptions[0].text,caption:document.getElementById('joint-end').textContent,marked:[...document.querySelectorAll('svg[data-joint]')].every(x=>x.querySelector('circle'))})"))
 ws.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1150,'deviceScaleFactor':1,'mobile':False})
 ws.js("document.getElementById('joint-choice').value='3';document.getElementById('joint-choice').dispatchEvent(new Event('change'));document.getElementById('joints').scrollIntoView()")
 shot=ws.call('Page.captureScreenshot',{'format':'png'});(OUT/'review_desktop_elbow.png').write_bytes(base64.b64decode(shot['data']))
 checks['broken_anchors']=ws.js("[...document.querySelectorAll('a[href^=\"#\"]')].map(a=>a.hash.slice(1)).filter(id=>!document.getElementById(id))")
 checks['errors']=[e for e in ws.events if e.get('method')=='Runtime.exceptionThrown'];checks['images']=ws.js('document.images.length')
 checks['no_overflow']=all(v['scrollWidth']<=v['width'] for v in checks['viewports']);checks['question']='What can improve clarity?'
 checks['html_sha256']=hashlib.sha256((ROOT/'docs/H84_BODY_REPAIR_EXPLAINED_2026-10-21.html').read_bytes()).hexdigest();checks['checked_at']=time.strftime('%Y-%m-%d %H:%M:%S')
finally:
 if ws:
  try:ws.send({'id':999999,'method':'Browser.close'});ws.s.close()
  except OSError:pass
 try:proc.wait(timeout=8)
 except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=4)
 for _ in range(20):
  try:
   assert profile.resolve().parent==Path(tempfile.gettempdir()).resolve() and profile.name.startswith('h84_body_report_')
   shutil.rmtree(profile);break
  except PermissionError:time.sleep(.2)
 checks['temporary_browser_profile_removed']=not profile.exists();(OUT/'browser_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks))
