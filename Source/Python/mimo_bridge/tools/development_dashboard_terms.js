/* Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. */
/* Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file. */
/* Clickable definitions. Text stays readable without relying on hover or abbreviations. */
'use strict';
const DASHBOARD_TERMS={
 'Controllability':'Agreement between predicted and actual movement direction and size. The comparison re-predicts the same body state using another body’s command. The recent statistical check decides whether this score may influence goal choice. It is not a success rate.',
 'Learning progress':'Recent prediction error compared with its longer-term average, after each error is divided by the error of predicting no change. Changing surroundings can also change this score, so improvement alone does not prove learning.',
 'Coherence':'A wellness channel whose input is a co-movement statistic. In: each model step, per body, the command (26 handles x 9 numbers) and proprioception (11 numbers per handle), each beside its values one model step earlier. Computed: for every handle, a running average of (size of the command\'s change x size of each proprioception number\'s change), each size divided by its own running root-mean-square; the handle\'s mean goes through a sigmoid, giving a number between 8 and 92 of 100. Out: with the extent setting, the mean of those numbers over the 26 handles and the tracked objects (28 of 100 for the handles at model step 455,800); with the certainty setting, how far the numbers sit from the middle of their range. The statistic has no time-lag bank, no sign and no comparison against movement without a command, so it counts how often command and proprioception change in the same model step, not whether the command caused the change. The run settings identify the selected formula.',
 'Capability':'Here this internal score means configured physical capacity, averaged across the body-capacity settings. It does not measure a learned skill.',
 'Integrity':'An internal body-safety score. It falls as skin-load pain or joint-limit penalties increase. It is not a general test of software correctness.',
 'Efficiency':'Here this internal score decreases with mechanical power and actuator torque. It measures physical effort, not learning speed.',
 'Touch-ranking accuracy':'How often a skin area about to be touched receives a higher predicted chance of touch than an area that stays untouched. A coin flip scores 50%; perfect ranking scores 100%. These are pairs of examples, not pairs of body parts touching.',
 'Encoder':'A network that turns a sense\'s input into the representation the model uses for prediction and planning.',
 'Stage':'In Model health, a numbered position within a software module. In the historical ability plan, a group of proposed abilities. The heading identifies which use applies; a stage number does not certify an acquired ability.',
 'Forward model':'The network that forecasts future observations from the current observations and the command.',
 'Attention':'A weighted selection of inputs used by a network. The attention-spread reading describes how concentrated that selection is, not whether it chose useful information.',
 'Token':'A list of learned numbers representing an input, such as a sense or the command, inside the model.',
 'Training update':'One round of experience used to adjust the shared model. Each simulated world supplies experience during the round. It is not one successful action.',
 'Motor synergy':'A small learned movement map. It represents a whole-body movement with a few numbers. Training the map and allowing it to direct the body are separate switches.',
 'Part-command':'One body part\'s share of a command: the 9 numbers for that part (target position, target rotation, radius, strength). A command holds one for each body part; their outcomes are not independent trials.',
 'Random-axis control':'A comparison using a randomly chosen axis (the line a body part turns around), keeping the requested turn angle. It is compared with the already observed movement, not a new simulated trial. Here, above chance means the measured result minus this geometric comparison; it does not prove useful control.',
 'Planner':'The part of the software that searches for a command: it scores candidate commands by how far the forward model\'s predicted outcome lies from a remembered goal, and keeps the best-scoring one. A planner being enabled does not show that its predictions or choices are good.',
 'Motor controller':'The network that proposes a command for each body part. Its proposal may be combined with a planned command or exploratory movement before execution.',
 'Command':'The numbers sent to the body: for each body part a target position, a target rotation, a radius and a strength (the code and older records call it `order`). The achieved movement can differ from the command because of physics and control limits.',
 'Model step':'One update of the model: it reads the senses and sends a command (the code and the records call it `step`). The counter continues across training restarts. More model steps do not by themselves show more ability.',
 'Run':'One launch of training with recorded settings. A restart begins a new run, also called a segment.',
 'Segment':'One continuous training run between launches. Charts mark its first recorded step with a vertical line.',
 'Lineage':'This model and the saved models it resumed from, traced back to its fresh start. Results from another lineage do not establish an ability in this model.',
 'Checkpoint':'A saved copy of the model and its training state at a particular step. Comparing saved copies can show what changed during training.',
 'Comparison':'A reference result measured alongside the model. The chart or test names the alternative, such as predicting no movement or using another body\'s command.',
 'Control':'In a test, a comparison condition used to check whether an effect is caused by the feature being tested. In “control on/off”, it means whether that feature may direct body movement.',
 'Goal line':'A stated target for a measurement. Reaching it is not an ability pass unless a test explicitly defines and checks that pass condition. “Unmeasured” means no justified target has been set.',
 'Floor':'The reference level named by that chart, often what knowing nothing or making no change would score. It is not necessarily the smallest possible value.',
 'Stand-in':'An indirect measurement used when the intended measurement is unavailable. It cannot establish the intended ability on its own.',
 'Frozen weights':'A comparison model whose learned numbers are held fixed during the measurement. Its errors can still change as the situation changes.',
 'Unit':'In Model health, a unit is one computational element inside a software network, not a muscle or body part. In a chart scale, units instead specify the measurement, such as degrees, seconds or bytes.',
 'Layer':'A group of computational units in the software network. A low response reading concerns that layer, not an anatomical part of the simulated body.',
 'Responding':'This monitor counts a unit when at least one sampled input leaves the nearly flat low end of its activation curve. At that low end, output changes by less than 1.2% of the input change. This does not measure whether the unit has learned a useful function.',
 'Warning line':'A monitoring threshold: fewer than 50 of every 100 units responding raises a health warning. This is not an ability pass/fail test, nor proof that the whole layer stopped working.',
 'Automatic repair':'For explicitly covered layers, the software monitors persistently unresponsive units and can reset those units. “Monitoring” does not mean a repair has already succeeded.',
 'Probe':'A controlled movement test on an identified saved model. It changes a specified input or command and compares predicted and actual outcomes. Its result belongs to the tested checkpoint.',
 'Body':'One of the simulated bodies training in parallel. Body 1 is the first scene. A reading averaged over bodies can differ from the single body visible in a camera.',
 'Skin patch':'One small sensor area on the simulated skin. Several patches can register a single physical contact.',
 'Report window':'The interval of training summarized by a record. Counts described as averages need not be whole numbers, and are not instantaneous camera readings.',
 'Touching body-part pair':'Two body parts touching each other. “Pairs per body” is averaged across the bodies and model steps in the report window, so a value such as 1.3 is possible.',
 'Posture':'The body\'s orientation. For the displayed roll angle: 0 degrees is on its back, 90 is on its side, and 180 is belly down. This is an observation, not proof of intentional rolling.',
 'Pose head':'The tenure pose head: a learned network that predicts, from the model step a command began and the command itself, how far each body part moves (its position, not its turn) over the command\'s whole tenure, a quarter of a simulated second.',
 'Wellness predictor':'The learned copy of wellness (the code\'s ApproximateEvaluator; WELLNESS_PREDICTOR in the component manifest): a network trained to copy five of the fixed wellness formulas from the tokens, so the planner can score imagined model steps, which have no skin or joint reading of their own.',
 'Prediction':'What the model forecasts before the outcome is observed. A useful prediction must beat the comparison named by its test.',
 'Prediction error':'The difference between a prediction and the observed outcome. Smaller is better only when compared on the same task, units and examples.',
 'Persistence':'The comparison that predicts the next observation will be unchanged. Beating it shows a useful forecast on those examples; it does not by itself establish a skill.',
 'Shuffled command':'Another body\'s command substituted while keeping the tested body\'s starting information. This tests whether knowing its own command improves the prediction (the records call it `shuffled order`).',
 'Wellness':'The model\'s internal scoring of conditions such as body integrity, prediction and contact. It guides choices; it is not a human assessment of health or consciousness.',
 'Tactile comfort':'The fixed skin-comfort score: 50% without contact or remaining load history, up to 85% from area-weighted pleasant contact near 5 kPa, reduced by the largest skin-patch aversion.',
 'Efference':'A prediction of what an outgoing command will do to the body or eyes. Its accuracy is checked against what actually happens.',
 'Body head':'One of the efference heads: from the command a body was sent, it predicts how far each body part moves and turns over the next 50 ms and over the whole 250 ms command.',
 'Gaze head':'One of the efference heads: from the eye command, it predicts how far the eyes turn over the next 50, 100 and 150 ms.',
 'Gaze copy':'Part of the efference heads: it adds the eye command\'s expected effect to the predicted next view, so the predicted view depends on where the eyes were told to look.',
 'Prediction head':'The network that turns the forward model\'s state into one sense\'s prediction of the next model step. There is one per predicted sense.',
 'Contact head':'The network that predicts, for each skin patch, whether it will be touched in the next 50 ms and whether by the body\'s own part or by an outside thing.',
 'Voice head':'A network that outputs 80 voice-control numbers. No lesson trains it, and it makes no sound in this world.',
 'Vision':'What the two eyes see: a wide view and a sharp centre for each eye, as pixels.',
 'Tactile':'The skin\'s sense of touch: up to 32 skin parts at once, each with its pressure, rub, slip, contact area, vibration and push.',
 'Audio':'What the two ears hear: how loud each of 256 pitch bands is at each ear, plus two numbers for the sound\'s direction.',
 'Proprioception':'Where each tracked body part is, how it is turned and how fast it moves.',
 'Motor (sense)':'The command the body was last given, beside the pose each part reached, read back as an input.',
 'Vestibular':'The body\'s load and balance in 29 numbers: which way gravity pulls on the head and how the head turns, the neck, which way is up for the trunk, what carries the weight, effort, joints at their limits and fatigue.',
 'Intermind':'A reserved input for other minds\' summaries, placeholder noise for now.',
 'Causal chain':'The recorded sequence from goal selection through predicted outcome, planned command and executed command. Recording the sequence does not prove that each link works well.',
 'Gate':'A rule that allows a measurement or contribution to count only after its evidence conditions are met. “Open” refers to that rule, not to a demonstrated ability.',
 'Attention spread':'How broadly an attention pool uses its inputs. Zero means concentrated on one input; 100% means evenly spread. There is no health warning threshold for this reading.',
 'Tenure':'The interval for which one command is held. A completed command is graded over this full interval, not over one camera frame.',
 'Reconstruction':'Reproducing a recorded observation or movement from a compressed representation. A good reconstruction does not prove that the model can cause that movement.',
 'Commit charge':'Memory Windows has promised to applications, backed by RAM or the page file. Near the commit limit, a new allocation can fail even if some RAM appears free.',
 'GPU':'The graphics processor used for model calculations. Its own memory is separate from the computer\'s main RAM.',
 'RAM':'The computer\'s physical working memory. It differs from committed memory, which can also be backed by the page file.',
 'GiB':'A binary memory unit: 1 GiB is 1,073,741,824 bytes. It is about 1.074 decimal GB.',
 'MiB':'A binary memory unit: 1 MiB is 1,048,576 bytes. There are 1,024 MiB in one GiB.',
 'GB':'A decimal memory unit: 1 GB is 1,000,000,000 bytes. See the chart definition for the unit used by that measurement.',
 'ms':'Milliseconds: 1,000 ms equals one second.',
 'Reflex':'Here, a command from the existing motor controller rather than the planner. This software label does not establish a biological reflex.',
 'Exploration':'Movement variation used to encounter new outcomes. Synergy exploration can coordinate that variation; it remains separate from planned control.',
 'Frame':'One camera image. A recording is a sequence of frames; a model step is a different event.',
 'Unmeasured':'No usable measurement is available for this field and scope. It means unknown, not zero and not failure.',
 'Gain of function':'An improvement in what the model can reliably do. A changed internal score or more movement is insufficient without an appropriate ability test.',
 'Scheduled comparison':'A test of a frozen saved copy of the model on physical goals, run by the comparison scheduler on its declared schedule of training updates. It records an outcome; it does not pass an ability or change a control switch.',
 'Starting controller':'The motor controller\'s own proposed command, before any planner search. The scheduled comparisons use it as the reference a planner must beat from the same saved state.',
 'Sample uncertainty':'How much a result could change with a different draw of the same bodies. A difference beyond the sample uncertainty has a paired 95% interval, from resampling the 8 bodies, that excludes zero.',
 'Held-out start':'A saved starting model step that was not used to set up or calibrate the test. Scores come from held-out starts.',
 'Auditor':'The test script. When the auditor chooses the commands, the test measures prediction, not the model\'s own choice.'
};
const TERM_ALIASES={'prediction heads':'Prediction head','motor token':'Motor (sense)','motor encoder':'Motor (sense)',synergy:'Motor synergy','held-out starts':'Held-out start','scheduled comparisons':'Scheduled comparison',step:'Model step',steps:'Model step','model steps':'Model step',runs:'Run',segments:'Segment',ancestry:'Lineage',checkpoints:'Checkpoint',comparisons:'Comparison',units:'Unit',layers:'Layer',probes:'Probe',bodies:'Body',patch:'Skin patch',patches:'Skin patch',commands:'Command',[`order`]:'Command',[`orders`]:'Command',[`shuffled order`]:'Shuffled command','shuffled-command control':'Shuffled command',predictions:'Prediction',gates:'Gate',frames:'Frame','window mean':'Report window','window means':'Report window'};
const termLookup=new Map(Object.keys(DASHBOARD_TERMS).map(k=>[k.toLowerCase(),k]));
termLookup.set('part-commands','Part-command');termLookup.set(`part-orders`,'Part-command');termLookup.set('body-part commands','Part-command');
termLookup.set('random-axis comparison','Random-axis control');termLookup.set('random-axis geometric control','Random-axis control');
for(const [alias,key] of Object.entries(TERM_ALIASES))termLookup.set(alias,key);
// “Run” can also be a verb. Link only the unambiguous phrase in prose.
termLookup.delete('run');termLookup.delete('runs');termLookup.set('training run','Run');
const escapeTerm=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
// A term inside a dotted code path (recon_decoders.vision.0) is not prose: splitting the path into a
// button left ".0" standing alone as a bare decimal. A term must not follow a word character or a dot,
// nor run on into one or into ".x"; a sentence's closing full stop still ends a term.
const termPattern=new RegExp('(?<![\\w.])('+[...termLookup.keys()].sort((a,b)=>b.length-a.length).map(escapeTerm).join('|')+')(?!\\w|\\.\\w)','gi');
function openTerm(key){
 const dialog=document.getElementById('term-dialog'),body=document.getElementById('term-definition');
 document.getElementById('term-title').textContent=key||'Terms and definitions';body.replaceChildren();
 if(key){const p=document.createElement('p');p.textContent=DASHBOARD_TERMS[key];body.append(p);}
 else{
  const input=document.createElement('input');input.type='search';input.placeholder='Find a term';input.setAttribute('aria-label','Find a definition');body.append(input);
  for(const name of Object.keys(DASHBOARD_TERMS).sort()){const d=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('p');s.textContent=name;p.textContent=DASHBOARD_TERMS[name];d.append(s,p);body.append(d);}
  input.addEventListener('input',()=>{for(const d of body.querySelectorAll('details'))d.hidden=!d.textContent.toLowerCase().includes(input.value.toLowerCase());});
 }
 dialog.showModal();
}
function defineText(root){
 const texts=[];
 const consider=n=>{if(n.nodeType===3&&n.parentElement&&!n.parentElement.closest('button,a,summary,script,style,code,pre,svg,select,dialog,[data-plain],[role="status"],#training-now,#connection'))texts.push(n);};
 consider(root);if(root.nodeType===1){const w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let n;while((n=w.nextNode()))consider(n);}
 for(const n of texts){const value=n.nodeValue;termPattern.lastIndex=0;const matches=[...value.matchAll(termPattern)];if(!matches.length)continue;
  const frag=document.createDocumentFragment();let i=0;for(const m of matches){frag.append(value.slice(i,m.index));const b=document.createElement('button'),key=termLookup.get(m[0].toLowerCase());b.className='term-help';b.type='button';b.textContent=m[0];b.dataset.term=key;b.setAttribute('aria-label','Definition: '+key);b.addEventListener('click',()=>openTerm(key));frag.append(b);i=m.index+m[0].length;}frag.append(value.slice(i));n.replaceWith(frag);
 }
}
// the header's "Terms and definitions" opens the tab of that name; a term button in the prose still opens its one
// definition in the dialog
document.getElementById('open-glossary').addEventListener('click',()=>{
 if(typeof showPage==='function'&&document.getElementById('terms')){showPage('terms');document.getElementById('tab-terms').focus();window.scrollTo(0,0);}
 else openTerm();
});
defineText(document.body);
new MutationObserver(records=>{for(const r of records)for(const n of r.addedNodes)if(n.isConnected)defineText(n);}).observe(document.body,{childList:true,subtree:true});

// ---------------------------------------------------------------- the Terms and definitions tab
// GET /api/names (tools/recovery_api.py names): every canonical name with its definition (the recovery directive of
// Oct 5, 2026, section 7B): the components of the component manifest under their ids, each with its plain name and its
// one sentence; the eight repair states, the defect severities, the test sections, outcomes, resource classes and
// process states, the machine modes, the scoreboard's stages and the two judgements of a reading; then the dashboard's
// other terms (DASHBOARD_TERMS above). The other names people have used for a component (its aliases) are searched
// and are never shown as a name. Every entry is open when the tab loads; the one button closes them all or opens them
// all, and its label says which it does next: the server's when_expanded label ("Collapse all") while every entry is
// open, its when_collapsed label ("Expand all") otherwise.
let namesData=null,namesError='',namesRequest=0;
async function loadTermsTab(){
 const mine=++namesRequest;
 try{const r=await fetch('/api/names',{cache:'no-store',signal:AbortSignal.timeout(30000)});if(mine!==namesRequest)return;
  if(r.status===404)throw Error('404');if(!r.ok)throw Error('the server answered '+r.status);namesData=await r.json();namesError='';}
 catch(e){if(mine!==namesRequest)return;namesError=e.message==='404'?'This tab needs the dashboard server restarted to load its new code.':'The names could not be read ('+e.message+').';}
 renderTermsTab();
}
function termsLabels(){const t=namesData?.toggle_labels||{};return {expanded:t.when_expanded||'Collapse all',collapsed:t.when_collapsed||'Expand all'};}
function termEntries(){return [...document.querySelectorAll('#terms-body details.term-entry')];}
function updateTermsToggle(){
 const all=termEntries(),open=all.length>0&&all.every(d=>d.open),b=document.getElementById('terms-toggle'),labels=termsLabels();
 b.textContent=open?labels.expanded:labels.collapsed;b.setAttribute('aria-expanded',String(open));b.disabled=!all.length;
}
function termNode(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}
// one entry: the name (an id in code type for a component), then its definition; `search` also holds what is searched
// and never shown (a component's aliases)
function termEntry(name,definition,{code=false,plain='',search='',more=[]}={}){
 const d=termNode('details','term-entry'),s=termNode('summary','term-name');d.open=namesData?.expanded_by_default!==false;
 if(code)s.append(termNode('code','',name));else s.append(termNode('span','',name));
 if(plain)s.append(termNode('span','term-plain',plain));
 d.append(s,termNode('p','term-definition',definition||'No definition is on record.'));
 for(const m of more)if(m)d.append(m);
 d.dataset.search=[name,plain,definition,search].join(' ').toLowerCase();
 d.addEventListener('toggle',updateTermsToggle);
 return d;
}
function termGroup(title,note,entries){
 const g=termNode('section','term-group'),h=termNode('h2','',title);g.append(h);
 if(note)g.append(termNode('p','small muted term-note',note));
 const list=termNode('div','term-list');for(const e of entries)list.append(e);g.append(list);
 return g;
}
function renderTermsTab(){
 const body=document.getElementById('terms-body'),status=document.getElementById('terms-status');if(!body)return;
 const n=namesData;
 if(!n){status.textContent=namesError||'Reading the names.';return;}
 status.textContent=(namesError?namesError+' ':'')+'Read '+(n.read_at_text||'')+' from the component manifest and the modules that compute with these names.';
 body.replaceChildren();
 const kinds=['model part','world','repair','tool'],comps=n.components||[];
 const explainer=String((typeof state!=='undefined'&&state?.explainer?.url)||'/files/Saved/mimo_bridge/explainer.html').split('#')[0];
 for(const kind of kinds){
  const mine=comps.filter(c=>c.kind===kind);if(!mine.length)continue;
  body.append(termGroup('Components: '+kind+' ('+mine.length+')',kind==='repair'?'A repair’s status is written as NAME: STATE with one of the eight repair states below.':'',
   mine.map(c=>{
    const link=termNode('a','term-link','Open '+c.id+' in the Architecture tab');link.href=explainer+'#t=arch&component='+encodeURIComponent(c.id);
    const facts=[];if((c.flags||[]).length)facts.push('Launch flags: '+c.flags.join(', ')+'.');
    return termEntry(c.id,c.definition,{code:true,plain:c.name,search:(c.aliases||[]).join(' '),more:[facts.length?termNode('p','small muted',facts.join(' ')):null,link]});
   })));
 }
 const named=(list,key='name')=>(list||[]).map(x=>termEntry(x[key],x.definition));
 if(n.components_error)body.append(termNode('p','error',n.components_error));
 body.append(termGroup('Repair states',[n.fixed_rule,n.status_form].filter(Boolean).join(' '),named(n.repair_states)));
 if((n.severity_meanings||[]).length)body.append(termGroup('Defect severities','',named(n.severity_meanings)));
 body.append(termGroup('Test sections','The seven sections of the Tests page, computed from the runner’s record.',named(n.test_sections)));
 body.append(termGroup('Test outcomes','The outcome of one run, written by the runner and by nothing else.',named(n.test_outcomes)));
 body.append(termGroup('Resource classes','',named(n.resource_classes)));
 body.append(termGroup('Process states of a running row','',named(n.process_states)));
 body.append(termGroup('Machine modes','The band under the tabs shows exactly one of these.',named(n.machine_modes)));
 body.append(termGroup('Stages of the gain-of-function scoreboard','',(n.scoreboard_stages||[]).map(s=>termEntry(s.plain_name||s.name,s.definition,{plain:s.name}))));
 if(typeof JUDGEMENT_TERMS!=='undefined'){
  const answers=n.reading_judgements||{};
  body.append(termGroup('The two judgements of a reading','',JUDGEMENT_TERMS.map(([name,definition],i)=>termEntry(name,definition+'. Answers: '+((i?answers.gof_material:answers.statistically_clear)||[]).join(', ')+'.'))));
 }
 body.append(termGroup('Other terms of the dashboard','',Object.keys(DASHBOARD_TERMS).sort().map(k=>termEntry(k,DASHBOARD_TERMS[k]))));
 filterTerms();updateTermsToggle();
}
function filterTerms(){
 const q=(document.getElementById('terms-search').value||'').trim().toLowerCase();let shown=0;
 for(const d of termEntries()){const hit=!q||d.dataset.search.includes(q);d.hidden=!hit;shown+=hit?1:0;}
 for(const g of document.querySelectorAll('#terms-body .term-group'))g.hidden=![...g.querySelectorAll('details.term-entry')].some(d=>!d.hidden);
 const all=termEntries().length;
 document.getElementById('terms-count').textContent=q?shown+' of '+all+' terms match':all+' terms';
}
document.getElementById('terms-search').addEventListener('input',filterTerms);
document.getElementById('terms-toggle').addEventListener('click',()=>{
 const all=termEntries(),open=all.length>0&&all.every(d=>d.open);
 for(const d of all)d.open=!open;
 updateTermsToggle();
});
if(location.hash==='#terms')loadTermsTab();
