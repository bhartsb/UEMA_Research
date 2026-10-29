/* Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. */
/* Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file. */
'use strict';
const $=id=>document.getElementById(id);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
let state=null,selectedAbility='wellness',selectedVideo=null,evidenceSignature='',lastApiAt=0,apiError=false;
const compact=matchMedia('(max-width:700px)'),tabs=[...document.querySelectorAll('[data-page]')];
// clock times in AM/PM with the zone ("Sep 24, 2026, 6:05 PM PDT"), never a 24-hour clock (operator 2026-09-23)
const dated=seconds=>new Date(seconds*1000).toLocaleString('en-US',{year:'numeric',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
const clockAt=seconds=>new Date(seconds*1000).toLocaleString('en-US',{month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
const plural=(n,word)=>n+' '+word+(n===1?'':'s');
const age=seconds=>{const s=Math.max(0,Math.floor(seconds));return s<60?plural(s,'second'):s<3600?plural(Math.floor(s/60),'minute'):plural(Math.floor(s/3600),'hour');};
const stepText=n=>Number.isFinite(n)?Math.round(n).toLocaleString():'—';
const UPGRADE_NAMES={1:'Trace choices and outcomes',2:'Value accurate control',3:'Value learning progress',4:'Balance needs and priorities',5:'Coordinate movements',6:'Predict self-touch',7:'Test eye control',8:'Test body control',9:'Test seeing and touching the body',10:'Add the vestibular input and test posture'};
const upgradeTitle=item=>UPGRADE_NAMES[item?.item]||item?.name||'Upgrade';

const PROOF_STATES={
 signal:{label:'Early signal',mark:'◐',meaning:'A prediction comparison helps, but the complete capability has not passed its test.'},
 concern:{label:'Investigating',mark:'!',meaning:'A measured weakness is being investigated. Open the capability to see the next test.'},
 untested:{label:'Not tested',mark:'○',meaning:'No reviewed test establishes this stage on the current model. This is an evidence gap, not a score of zero.'},
 passed:{label:'Passed',mark:'✓',meaning:'The named capability passed its declared test; the linked record identifies its model and test conditions.'}
};
const HAND_CAPABILITIES=[
 ['Open and close','Extend the fingers and bend them through their useful ranges, including while the wrist moves.'],
 ['Move one finger','Move a selected finger and measure unintended neighbouring movement. The simulated tendons couple some knuckles, so complete independence is not always physically possible.'],
 ['Spread and gather','Move fingers apart and back together where the simulated joints permit.'],
 ['Use the thumb','Bring the thumb toward the fingertips and control thumb bending.'],
 ['Oppose and return the thumb','Turn the thumb to face a fingertip, make controlled contact, then separate.'],
 ['Shape the palm','Cup and flatten the hand within the simulated joints’ available movement.'],
 ['Bend and angle the wrist','Move the wrist forward, back and sideways while keeping the chosen grip.'],
 ['Turn the forearm','Turn the palm up or down while preserving the finger goal.'],
 ['Reach and orient','Bring the hand to an object with the palm and fingers positioned to use it.'],
 ['Touch, press and point','Select a contacting finger or hand surface and control the contact force.'],
 ['Whole-hand grasp','Close around an object, sustain contact and support its weight.'],
 ['Pinch','Hold a small object between thumb and finger.'],
 ['Adjust a grip','Change finger force or placement when an object slips or its load changes.'],
 ['Release and place','Open deliberately to let go or put an object in a chosen location.'],
 ['Turn and reposition','Change an object’s orientation or its position inside the hand.'],
 ['Transfer hands','Pass an object from one hand to the other without dropping it.'],
 ['Coordinate both hands','Hold an object with one hand while the other acts on it.'],
 ['Use objects','Operate a button, container or simple tool through a chosen sequence.']
];
document.querySelectorAll('[data-atlas-target]').forEach(b=>b.addEventListener('click',()=>{const target=document.querySelector(b.dataset.atlasTarget);target.scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});const focus=target.querySelector('button,summary,a');if(focus)focus.focus({preventScroll:true});}));
function goalDialog(title,paragraphs,source){
 const dialog=$('term-dialog'),body=$('term-definition');$('term-title').textContent=title;body.replaceChildren();
 for(const p of paragraphs.filter(Boolean))body.append(el('p','',p));
 if(source){const a=el('a','','Open the evidence and method ↗');a.href=source;body.append(a);}
 dialog.showModal();
}
function goalSource(parent,url,label='Evidence and test details ↗'){
 if(!url)return;const a=el('a','goal-source',label);a.href=url;parent.append(a);
}
// every catalogue card carries the server's freshness (bind_card): when its text was checked against its
// sources, which results its numbers come from, and since when it is out of date (a newer result arrived)
function freshText(row){const f=row?.freshness;if(!f)return null;return (f.out_of_date?f.stale_text+' ':'As of: ')+f.text;}
function cardDate(row,compact=false){
 const f=row?.freshness;if(!f)return null;const p=el('p','card-date'+(f.out_of_date?' stale':''));p.title=freshText(row);
 if(f.out_of_date)p.append(el('strong','','Out of date'+(f.since_text?' since '+f.since_text:'')+'.'),document.createTextNode(compact?'':' '+(f.stale_reason?f.stale_reason.charAt(0).toUpperCase()+f.stale_reason.slice(1):'')+' '+f.text));
 else p.textContent=compact?'As of '+(f.checked_text||'an unrecorded time'):'As of: '+f.text;
 return p;
}
const ATLAS_ICONS={
 eyes:['M3 12C7 4 17 4 21 12C17 20 7 20 3 12Z','M15 12A3 3 0 1 1 9 12A3 3 0 1 1 15 12'],
 reach:['M3 19L8 11L14 14L19 5','M14 5H19V10','M7 21H3V17'],
 hands:['M7 13V6Q7 4 9 6V11V3Q11 1 11 3V11V4Q13 2 13 4V11V6Q15 4 15 6V13L18 10Q21 9 20 12L16 20H8L4 14Q3 11 5 12Z'],
 touch:['M4 17L10 11L14 15L18 11','M16 4A4 4 0 1 1 16 12A4 4 0 1 1 16 4','M3 21H20'],
 roll:['M5 8A8 8 0 1 1 4 15','M3 4L5 8L9 6','M8 14L11 9L16 13L13 17Z'],
 self_directed:['M20 12A8 8 0 1 1 12 4','M15 3H21V9','M11 13L21 3','M5 20L8 17']
};
function atlasIcon(name){
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');svg.classList.add('atlas-icon');
 for(const d of ATLAS_ICONS[name]||ATLAS_ICONS.self_directed){const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',d);svg.append(path);}return svg;
}
function atlasRecordDialog(row,title,stageIndex){
 const data=state.goal_catalog,stage=stageIndex===undefined?null:data.stages[stageIndex],available=row.source_available!==false&&Boolean(row.source_url);
 const paragraphs=[row.definition||row.detail];
 if(stage){const proof=PROOF_STATES[available?row.stage_states?.[stageIndex]||'untested':'untested'];paragraphs.push(row.item?'This stage concerns the abilities supported by this upgrade. '+stage.definition:stage.definition,proof.label+': '+(available?(row.stage_evidence?.[stageIndex]||row.evidence||'No reviewed evidence is attached for this stage.'):'No reviewed evidence source is attached for this stage.'));}
 else{if(row.physical_detail)paragraphs.push('Physical test: '+row.physical_detail);if(row.evidence)paragraphs.push(row.evidence);}
 paragraphs.push('Next action: '+(row.next||'The next test has not been recorded.'),freshText(row));
 goalDialog(title+(stage?' · '+stage.title:''),paragraphs,available?row.source_url:null);
 if(row.upgrades?.length){const nav=el('div','dialog-connections');nav.append(el('p','','Connected upgrades'));for(const number of row.upgrades){const item=state.gain_of_function?.items.find(x=>x.item===number);if(!item)continue;const b=el('button','goal-chip',number+'. '+upgradeTitle(item));b.type='button';b.addEventListener('click',()=>{$('term-dialog').close();const target=$('upgrade-atlas').querySelector('[data-upgrade="'+number+'"]');target.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});target.querySelector('button').focus({preventScroll:true});});nav.append(b);}$('term-definition').append(nav);}
 if(row.id==='hands'){const b=el('button','text-button','Explore all 18 hand movements →');b.type='button';b.addEventListener('click',()=>{$('term-dialog').close();$('hand-program').scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});$('hand-catalog').querySelector('button').focus({preventScroll:true});});$('term-definition').append(b);}
}
function atlasProgress(row,title){
 const strip=el('div','atlas-progress');strip.setAttribute('aria-label','Evidence stages for '+title);
 const labels=['Predict','Choose','Repeat','Transfer'];
 for(const [i,stage]of state.goal_catalog.stages.entries()){
  const available=row.source_available!==false&&Boolean(row.source_url),key=available?row.stage_states?.[i]||'untested':'untested',proof=PROOF_STATES[key]||PROOF_STATES.untested;
  const b=el('button','atlas-proof '+key);b.type='button';b.setAttribute('aria-label',title+' · '+stage.title+': '+proof.label);b.title=stage.title+': '+proof.label;
  b.append(el('span','atlas-proof-mark',proof.mark),el('span','atlas-proof-label',labels[i]));b.addEventListener('click',()=>atlasRecordDialog(row,title,i));strip.append(b);
 }return strip;
}
function renderAttention(data){
 const container=$('goal-attention');container.replaceChildren();
 // present status first: notes written before production resumed move below the live work and
 // the scheduled comparisons; current notes keep their place above them
 if(data.as_of){if(container.previousElementSibling!==$('automatic-checks'))$('automatic-checks').after(container);}
 else if(container.nextElementSibling!==$('goal-next'))$('goal-next').before(container);
 // written before production resumed (server's as_of): the cards keep their words, muted, under one date line
 if(data.as_of&&(data.attention||[]).length)container.append(el('p','catalog-date stale attention-date',data.as_of));
 for(const item of data.attention||[]){
  const card=el('article','attention-card '+(['critical','overdue','active'].includes(item.tone)?item.tone:'active')+(data.as_of?' dated':''));
  const head=el('div','attention-top');head.append(el('span','attention-symbol',item.tone==='overdue'?'↻':item.tone==='critical'?'!':'→'),el('span','attention-status',item.status));
  const b=el('button','attention-title',item.title);b.type='button';b.addEventListener('click',()=>goalDialog(item.title,[data.as_of?data.as_of+'.':null,item.summary,item.detail,'Next action: '+item.next,freshText(item)],item.source_available===false?null:item.source_url));
  card.append(head,b,el('p','',item.summary));const dated=cardDate(item);if(dated)card.append(dated);container.append(card);
 }container.hidden=!container.children.length;
}
// every hand-written catalogue section carries the date its notes were written
function renderCatalogDates(data){
 for(const note of document.querySelectorAll('[data-catalog-note]')){note.textContent=data?.as_of||(data?.written_text?'Written '+data.written_text:'');note.hidden=!note.textContent;note.classList.toggle('stale',Boolean(data?.as_of));}
}
function renderAbilityAtlas(data){
 for(const legend of document.querySelectorAll('[data-atlas-legend]')){legend.replaceChildren();for(const [name,proof]of Object.entries(PROOF_STATES)){const b=el('button','proof-key '+name);b.type='button';b.append(el('span','proof-mark',proof.mark),document.createTextNode(proof.label));b.addEventListener('click',()=>goalDialog(proof.label,[proof.meaning]));legend.append(b);}}
 const key=$('atlas-stage-key');key.replaceChildren();
 for(const [i,stage]of data.stages.entries()){const b=el('button','atlas-stage-definition');b.type='button';b.append(el('span','stage-number',String(i+1)),el('strong','',stage.title));b.addEventListener('click',()=>goalDialog(stage.title,[stage.definition]));key.append(b);}
 const abilities=$('ability-map');abilities.replaceChildren();
 for(const row of data.capabilities){const card=el('article','atlas-ability '+row.id);const title=el('button','atlas-ability-title');title.type='button';title.append(atlasIcon(row.id),el('strong','',row.title));title.addEventListener('click',()=>atlasRecordDialog(row,row.title));card.append(title,atlasProgress(row,row.title));const dated=cardDate(row,true);if(dated)card.append(dated);abilities.append(card);}
 const counts=$('atlas-counts');counts.replaceChildren();const chosen=data.capabilities.filter(r=>r.source_available&&r.stage_states[1]==='passed').length;
 counts.append(el('strong','',chosen+' / '+data.capabilities.length),el('span','','chosen abilities demonstrated'));
 renderHandAtlas(data);renderUpgradeAtlas(data);renderAttention(data);
}
function renderHandDrawing(groups){
 const holder=$('hand-anatomy');holder.replaceChildren();const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 260 320');svg.setAttribute('aria-hidden','true');
 const paths=[['hand-outline','M84 292L76 247Q62 223 49 207L22 172Q13 158 23 149Q33 142 44 153L70 181L54 93Q51 78 63 76Q76 76 80 91L93 147L88 47Q87 32 100 31Q112 31 114 45L121 137L127 30Q128 16 141 18Q153 20 151 35L145 139L162 54Q165 41 177 45Q189 49 185 63L174 182Q181 217 163 240L162 292Z'],['hand-bone','M106 274L99 220L83 181L66 92M129 274L128 212L111 162L101 48M151 273L153 216L139 164L140 34M155 214L162 173L176 59M100 220L64 200L31 161'],['hand-motion','M31 65Q96 -3 187 16M178 8L187 16L175 23M198 229Q224 203 225 177M218 184L225 177L232 185']];
 for(const [cls,d]of paths){const p=document.createElementNS(svg.namespaceURI,'path');p.setAttribute('d',d);p.setAttribute('class',cls);svg.append(p);}
 for(const [x,y]of [[83,181],[76,139],[111,162],[107,102],[139,164],[140,96],[162,173],[169,116],[64,200],[128,212]]){const c=document.createElementNS(svg.namespaceURI,'circle');c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r','4');c.setAttribute('class','hand-joint');svg.append(c);}holder.append(svg);
 const list=el('div','hand-group-nav');for(const [index,name]of groups.entries()){const b=el('button','hand-group-link');b.type='button';b.append(el('span','',String(index+1)),document.createTextNode(name));b.addEventListener('click',()=>{const target=$('hand-group-'+index);target.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'center'});target.querySelector('button').focus({preventScroll:true});});list.append(b);}holder.append(list);
}
function renderHandAtlas(data){
 const fallback=HAND_CAPABILITIES.map(([title,definition],i)=>({id:'hand-'+i,title,definition,group:i<6?'Fingers and thumb':i<10?'Wrist, reach and touch':i<15?'Grasp and object control':'Two hands and object use',stage_states:['untested','untested','untested','untested'],source_url:data.hand_program_url,source_available:Boolean(data.hand_program_url),next:'No action-specific learned-ability test is recorded yet.'}));
 const rows=data.hand_actions?.length?data.hand_actions:fallback,groups=[...new Set(rows.map(r=>r.group))],catalog=$('hand-catalog');catalog.replaceChildren();renderHandDrawing(groups);
 for(const [groupIndex,group]of groups.entries()){const section=el('section','hand-action-group');section.id='hand-group-'+groupIndex;const h=el('h3');h.append(el('span','hand-group-number',String(groupIndex+1)),document.createTextNode(group));section.append(h);const grid=el('div','hand-action-grid');
  for(const row of rows.filter(r=>r.group===group)){const card=el('article','hand-action-card');card.dataset.handAction=row.id;const b=el('button','hand-skill');b.type='button';b.append(el('span','hand-action-number',String(rows.indexOf(row)+1).padStart(2,'0')),el('strong','',row.title));b.addEventListener('click',()=>atlasRecordDialog(row,row.title));card.append(b);
   const physical=el('button','physical-status',row.source_available===false?'Physical source unavailable':row.physical_status||'Physical result not linked');physical.type='button';physical.addEventListener('click',()=>goalDialog(row.title+' · Physical controls',[row.definition,row.source_available===false?'The linked physical result is unavailable.':row.physical_detail||'No action-specific physical calibration is attached.','These tests use researcher-supplied commands. Learned choice is scored separately.','Next action: '+(row.next||'No next action is recorded.'),freshText(row)],row.source_available===false?null:row.source_url));card.append(physical,atlasProgress(row,row.title));const dated=cardDate(row,true);if(dated)card.append(dated);grid.append(card);
  }section.append(grid);catalog.append(section);
 }
}
function renderUpgradeAtlas(data){
 const board=$('upgrade-atlas');board.replaceChildren();
 for(const item of state.gain_of_function?.items||[]){const record=data.upgrade_progress?.find(r=>r.item===item.item),row=record||{stage_states:['untested','untested','untested','untested'],definition:'No reviewed ability-stage mapping is attached to this upgrade.',next:'Open the recorded measurements and review its linked abilities.'};const card=el('article','upgrade-map-card');card.dataset.upgrade=item.item;
  const b=el('button','upgrade-map-title');b.type='button';b.append(el('span','upgrade-number',String(item.item).padStart(2,'0')),el('strong','',upgradeTitle(item)));b.addEventListener('click',()=>atlasRecordDialog(row,upgradeTitle(item)));card.append(b,atlasProgress(row,upgradeTitle(item)));
  const linked=el('div','upgrade-ability-links');for(const ability of data.capabilities.filter(c=>c.upgrades.includes(item.item))){const link=el('button','',ability.title);link.type='button';link.addEventListener('click',()=>atlasRecordDialog(ability,ability.title));linked.append(link);}card.append(linked);
  const measurement=el('button','text-button','Measurements and controls ↗');measurement.type='button';measurement.addEventListener('click',()=>openUpgrade(item.item));card.append(measurement);const dated=cardDate(record,true);if(dated)card.append(dated);board.append(card);
 }
}
let goalSignature='',selectedGoal=null;
function selectGoal(id){
 const data=state?.goal_catalog,g=data?.capabilities.find(c=>c.id===id);if(!g)return;
 $('capability-matrix').closest('details').open=true;
 selectedGoal=selectedGoal===id?null:id;
 for(const b of $('capability-matrix').querySelectorAll('[data-goal]'))b.setAttribute('aria-expanded',String(b.dataset.goal===selectedGoal));
 const panel=$('capability-detail');panel.hidden=!selectedGoal;panel.replaceChildren();if(!selectedGoal)return;
 panel.append(el('h3','',g.title),el('p','',g.detail));
 if(!g.source_available){panel.append(el('p','','Evidence file unavailable. Capability claims are withheld.'));return;}
 if(g.evidence)panel.append(el('p','',g.evidence));
 const next=el('p');next.append(el('strong','','Next test: '),document.createTextNode(g.next));panel.append(next);
 const connected=el('div','goal-connections');connected.append(el('span','small muted','Supported by: '));
 for(const number of g.upgrades){const item=state.gain_of_function?.items.find(x=>x.item===number);const b=el('button','goal-chip',item?upgradeTitle(item):'Upgrade '+number);b.type='button';b.addEventListener('click',()=>openUpgrade(number));connected.append(b);}panel.append(connected);
 const dated=cardDate(g);if(dated)panel.append(dated);goalSource(panel,g.source_url);if(g.id==='hands'&&data.hand_program_url)goalSource(panel,data.hand_program_url,'Full hand programme ↗');
}
function openUpgrade(number){
 $('gof-overview').open=true;const card=$('gof-items').querySelector('[data-item="'+number+'"]');if(!card)return;
 card.querySelector('.gof-reading').open=true;card.scrollIntoView({block:'center',behavior:'smooth'});card.tabIndex=-1;card.focus({preventScroll:true});
}
function automaticChecksSummary(now=Date.now()/1000){
 const w=state?.automatic_ability_checks;if(!w?.timestamp)return 'Automatic checks: awaiting worker report';
 if(apiError||now-lastApiAt>75)return 'Automatic checks: current status unavailable';
 if(now-w.timestamp>120)return 'Automatic checks: worker report overdue';
 const tasks=Object.values(w.tasks||{}),active=tasks.find(t=>['starting','running','process identity unavailable'].includes(t.status));
 // a failed round of a mechanism that steers production keeps training paused for review (the scheduler's hold)
 const review=tasks.find(t=>t.status==='held for review');
 if(w.comparison_window?.state==='held for review')return 'Automatic checks: training held for review — '+(review?.title||'a comparison')+' failed its declared comparison';
 // the scheduler's warnings (an unmeasured mechanism that steers production, a test running far too long) come
 // before a running test's name, which they may concern
 const warned=tasks.find(t=>t.warning);
 if(warned)return 'Automatic checks: WARNING — '+warned.warning;
 if(active)return 'Automatic check: '+(active.title||'Physical comparison')+' — '+active.status;
 if(w.comparison_window?.state==='saving')return 'Automatic checks: saving training to free memory for comparisons';
 if(review)return 'Automatic checks: '+review.title+' failed its declared comparison and awaits review; the next saved pause holds training';
 const failed=tasks.filter(t=>['failed','repair needed','memory capacity needs repair'].includes(t.status));
 if(failed.length)return 'Automatic checks: '+failed.length+' comparison'+(failed.length===1?' needs':'s need')+' attention';
 const due=tasks.map(t=>t.due_step).filter(Number.isFinite),next=due.length?Math.min(...due):null;
 if(w.state==='production paused for repair')return 'Automatic checks: paused with training'+(next===null?'':' · next at step '+stepText(next));
 if(w.comparison_window?.state==='ready')return 'Automatic checks: saved comparison window';
 // results and comparisons are counted from the receipts (the server's ability_results)
 const r=state?.ability_results,done=Number.isFinite(r?.completed)?plural(r.completed,'result')+' from '+plural(r.comparisons,'comparison'):tasks.filter(t=>t.completed_step!==undefined).length+' comparisons with results';
 // a held comparison (the server's with_holds) has no due step; the summary says how many are held
 const held=tasks.filter(t=>t.status==='held').length;
 return 'Automatic checks: '+done+(next===null?'':' · next at step '+stepText(next))+(held?' · '+held+' held':'');
}
function renderAutomaticChecks(){
 const panel=$('automatic-checks'),opened=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.dataset.result));panel.replaceChildren();const b=el('button','automatic-checks-button',automaticChecksSummary());b.type='button';b.addEventListener('click',()=>{
  const r=state?.ability_results||{},rows=r.rows||[],every=rows.map(x=>x.every_steps).find(Number.isFinite);
  // frequent comparisons repeat on their cadence; triggered ones run on a declared trigger and at least every every_steps
  const often=rows.filter(x=>x.cadence!=='triggered'&&Number.isFinite(x.every_steps)),triggered=rows.filter(x=>x.cadence==='triggered'&&Number.isFinite(x.every_steps));
  const cadence=triggered.length?'; '+(often.length?often.map(x=>x.title).join(', ')+' every '+stepText(often[0].every_steps)+' training updates; ':'')+'the others when one of their declared triggers fires (a code or flag change, a result crossing its line, a live gauge moving) and at least every '+stepText(triggered[0].every_steps):(Number.isFinite(every)?'; each repeats every '+stepText(every)+' training updates':'');
  const w=state?.automatic_ability_checks||{},paragraphs=['These frozen-model comparisons test body control, coordinated movement, chosen self-touch, eye predictions and finger following. A finished test records an outcome; it does not automatically pass an ability or enable a control switch.',(Number.isFinite(r.first_step)?'The first ran at step '+stepText(r.first_step)+cadence+'. ':'')+'The worker retries an interrupted test once and records persistent failures. If training leaves too little memory, it requests a complete saved pause, runs the comparisons and lets the existing supervisor resume. A repair or operator stop prevents automatic resume. A failed round of a comparison that steers production holds training for review.'];
  for(const t of Object.values(w.tasks||{}))paragraphs.push((t.title||'Comparison')+': '+(t.status||'scheduled')+'. '+(t.completed_step!==undefined?'Last completed checkpoint: '+stepText(t.completed_step)+'. ':'')+(Number.isFinite(t.due_step)?'Next due checkpoint: '+stepText(t.due_step)+'. ':'')+(t.reason||''));
  if(w.comparison_window?.reason)paragraphs.push(w.comparison_window.reason);
  paragraphs.push(w.timestamp?'Worker last reported '+dated(w.timestamp)+'.':'The worker has not reported yet.');
  goalDialog('Automatic physical comparisons',paragraphs,'/files/docs/gain_of_function_v3/ability_rechecks.json');
  if(w.timestamp)goalSource($('term-definition'),'/files/Saved/mimo_bridge/ability_rechecks/status.json','Open the task status and result paths');
 });panel.append(b);
 // the scheduler's warnings stay on the panel, above the results, until their cause clears
 for(const text of state?.automatic_ability_checks?.warnings||[])panel.append(el('p','ability-results-warning','Warning: '+text));
 // the newest completed result of each comparison, built by the server from its receipt and result file
 const rows=state?.ability_results?.rows||[];if(!rows.length)return;
 if(state.ability_results.lead)panel.append(el('p','ability-results-lead',state.ability_results.lead));
 const list=el('div','ability-results');
 for(const row of rows){
  const item=el('article','ability-result'),L=row.latest,head=el('div','ability-result-head');
  head.append(el('strong','',row.title),el('span','ability-result-when',L?'Checkpoint '+stepText(L.step)+' · finished '+L.when:'No completed result yet'));item.append(head);
  if(L)item.append(el('p','ability-result-headline',L.headline));
  if(row.next)item.append(el('p','ability-result-next',row.next));
  if(L){const d=el('details');d.dataset.result=row.id;d.open=opened.has(row.id);d.append(el('summary','','Method, earlier rounds and limits'));
   for(const text of [L.detail,row.trend,L.limit])if(text)d.append(el('p','small',text));
   const a=el('a','small','Open this result file ↗');a.href=L.url;d.append(a);item.append(d);}
  list.append(item);
 }panel.append(list);
}
function renderGoalProgramme(){
 const data=state.goal_catalog,signature=JSON.stringify([data,state.gain_of_function?.items.map(i=>[i.item,i.name]),state.ability_results]);
 if(signature===goalSignature)return;goalSignature=signature;
 if(!data){$('goal-title').textContent='Capability programme unavailable';$('goal-summary').textContent='The saved research programme could not be read. Live training details remain below.';for(const id of ['goal-next','automatic-checks','goal-highlights','goal-legend','capability-matrix','capability-detail','goal-investigations','goal-events','hand-catalog','upgrade-goal-map','goal-attention','ability-map','atlas-counts','atlas-stage-key','upgrade-atlas'])$(id).replaceChildren();$('goal-provenance').textContent='';renderCatalogDates(null);return;}
 $('goal-title').textContent=data.title;$('goal-summary').textContent=data.summary;$('goal-summary').hidden=!data.summary;$('goal-spec').href=data.source_url;
 // the hand-written notes carry the time they were written; notes older than production's newest state say so
 const provenance=$('goal-provenance');provenance.replaceChildren(document.createTextNode('Research notes written '+(data.written_text||data.reviewed_at)+' · Saved tests are dated below. The training status above is live.'));
 if(data.staleness)provenance.append(el('span','catalog-stale',data.staleness));provenance.classList.toggle('stale',Boolean(data.staleness));
 renderCatalogDates(data);
 const flow=$('goal-next');flow.replaceChildren();const a=data.activity;
 if(a){
  const first=el('article','active');first.append(el('span','flow-label','Now'),el('strong','',a.title));
  const status=a.kind==='production'?(a.status_label||'Reading the production state'):a.kind==='repair'?(a.work_status_label||a.status_label||'Production paused; repair comparisons in progress'):a.state==='milestone_due'?'Recovery check due at the next saved checkpoint':a.state==='paused'?'Training paused; progress saved':a.state==='stopped'?'Test stopped for correction':a.state==='starting'?(a.kind==='fixture'?'Preparing the private object fixture':'Restoring the saved private worlds'):a.state==='waiting'?'Waiting for an experiment report':a.state==='stale'?(a.kind==='training'?'Training report overdue':'Test update overdue'):a.state==='complete'?a.result_note:a.done+' of '+a.total+' '+(a.progress_unit||a.unit||'test updates recorded');const statusText=el('span','flow-status',status);statusText.id='activity-status';first.append(statusText);
  if((a.state==='running'||(a.kind==='production'&&a.state==='comparisons'))&&Number.isFinite(a.total)&&Number.isFinite(a.done)&&a.total>0){const progress=el('progress');progress.id='activity-progress';progress.max=a.total;progress.value=a.done;progress.setAttribute('aria-label',a.progress_unit||a.unit||'Completed test updates');first.append(progress);if(Number.isFinite(a.seconds_remaining)){const remaining=el('span','flow-status','About '+age(a.seconds_remaining)+' remaining');remaining.id='activity-remaining';first.append(remaining);}}
  // production: the supervisor's run and the comparison scheduler, read live by the server
  const trainingKind=['training','production'].includes(a.kind);
  const info=el('button','text-button',trainingKind?'Progress and timing':a.kind==='repair'?'Work and timing':'Test and timing');info.type='button';info.addEventListener('click',()=>goalDialog(a.title,[a.purpose,trainingKind?'The update count schedules a physical recheck; it does not mean a skill has been learned.':['experiment','qualification'].includes(a.kind)?'Qualification means testing the final repaired code on a copy of all eight saved worlds before restarting production. It checks state preservation, learning updates and model responses.':null,!apiError&&Date.now()/1000-lastApiAt<75&&Date.now()/1000-a.recorded_at<600&&Number.isFinite(a.seconds_remaining)?'Estimated remaining time: '+age(a.seconds_remaining)+'. '+a.estimate_basis:a.state==='milestone_due'?'The training milestone is reached. The saved-checkpoint comparison is now due.':a.state==='stopped'?a.result_note:a.state==='paused'?'The estimate resumes when training reports new progress.':a.state==='complete'?'This test has finished.':a.state==='comparisons'?'Training resumes after this window’s comparisons finish; no finishing time is estimated.':a.state==='due'?'The comparison waits for a complete saved checkpoint or free memory.':trainingKind?'A training estimate needs a current measured update rate.':a.kind==='repair'?'Repair completion has no measured time estimate. Finished comparisons and verification determine the next step.':'A completion estimate needs a current measured pace for this test.',a.recorded_at?(trainingKind?'Last training report: ':a.kind==='repair'?'Latest source record: ':'Last test report: ')+dated(a.recorded_at):'No report has been written yet.']));first.append(info);flow.append(first);
  for(const [label,title,detail]of [['Next',a.next,a.next_detail],['Then',a.then,a.then_detail]]){if(!title)continue;const card=el('article'),b=el('button','flow-action',title+' ⓘ');
   if(a.kind==='production'){const key=label==='Next'?'next':'then',remaining=a[key+'_remaining_updates'],due=a[key+'_due_at'];b.addEventListener('click',()=>goalDialog(title,[detail,a[key+'_calculation']]));b.type='button';card.append(el('span','flow-label',label),b);if(Number.isFinite(remaining)){const when=el('span','flow-status activity-due',stepText(remaining)+' training updates away'+(Number.isFinite(due)?' · about '+clockAt(due):''));card.append(when);}if(a[key+'_overdue'])card.append(el('span','flow-status schedule-gap',a[key+'_overdue']));if(a.scheduler_overdue)card.append(el('span','flow-status schedule-gap',a.scheduler_status||'Scheduler status unavailable'));flow.append(card);continue;}b.type='button';b.addEventListener('click',()=>goalDialog(title,[detail,...(label==='Next'&&a.next_estimate_basis?[a.next_estimate_basis]:[]),...(label==='Then'&&!['training','repair'].includes(a.kind)&&a.recovery?[a.recovery.meaning===detail?null:a.recovery.meaning,Number.isFinite(a.recovery.seconds_estimate)?'Training-time estimate: remaining updates divided by the production rate recorded '+dated(a.recovery.rate_recorded_at)+'. Setup, saving and ability tests add time. The estimate updates as production reports arrive.':null]:[])]));card.append(el('span','flow-label',label),b);if(label==='Next'&&Number.isFinite(a.next_seconds_estimate))card.append(el('span','flow-status','About '+age(a.next_seconds_estimate)+(a.next_estimate_includes_setup?' + review':' + setup and review')));if(label==='Next'&&a.kind==='repair'&&!Number.isFinite(a.next_seconds_estimate))card.append(el('span','flow-status','Duration not measured yet'));if(label==='Then'&&['training','repair'].includes(a.kind))card.append(el('span','flow-status','Duration not measured yet'));if(label==='Then'&&!['training','repair'].includes(a.kind)&&Number.isFinite(a.recovery?.remaining_updates)){card.append(el('span','flow-status',stepText(a.recovery.remaining_updates)+' updates to the recheck'));if(Number.isFinite(a.recovery.seconds_estimate))card.append(el('span','flow-status','About '+age(a.recovery.seconds_estimate)+' of training'));}flow.append(card);}
 }
 renderAutomaticChecks();
 const highlights=$('goal-highlights');highlights.replaceChildren();
 // a tile tied to a scheduled comparison (server: link_highlights) carries its newest round and time
 for(const h of data.highlights){const article=el('article','goal-highlight '+h.tone);article.append(el('h3','',h.title),el('strong','highlight-value',h.source_available?h.value:'Source unavailable'));if(h.live&&h.scope)article.append(el('span','highlight-scope',h.scope));
  const dated=cardDate(h,true);if(dated)article.append(dated);
  if(h.source_available){const b=el('button','highlight-details','Comparison and next step ↗');b.type='button';b.addEventListener('click',()=>goalDialog(h.title,[h.meaning,h.live?h.scope:null,freshText(h)],h.source_url));article.append(b);}highlights.append(article);}
 const legend=$('goal-legend');legend.replaceChildren();for(const [key,proof]of Object.entries(PROOF_STATES)){const b=el('button','proof-key '+key);b.type='button';b.append(el('span','proof-mark',proof.mark),document.createTextNode(proof.label));b.addEventListener('click',()=>goalDialog(proof.label,[proof.meaning]));legend.append(b);}
 const table=$('capability-matrix');table.replaceChildren();const head=el('thead'),hr=el('tr');hr.append(el('th','','Capability'));
 for(const stage of data.stages){const th=el('th'),b=el('button','stage-definition',stage.title+' ⓘ');b.type='button';b.addEventListener('click',()=>goalDialog(stage.title,[stage.definition]));th.append(b);hr.append(th);}head.append(hr);table.append(head);
 const body=el('tbody');for(const g of data.capabilities){const tr=el('tr'),th=el('th'),b=el('button','capability-name',g.title);b.type='button';b.dataset.goal=g.id;b.setAttribute('aria-expanded',String(selectedGoal===g.id));b.setAttribute('aria-controls','capability-detail');b.addEventListener('click',()=>selectGoal(g.id));th.append(b);tr.append(th);
  g.stage_states.forEach((s,i)=>{if(!g.source_available)s='untested';const proof=PROOF_STATES[s],td=el('td'),button=el('button','proof-cell '+s);button.type='button';button.setAttribute('aria-label',g.title+': '+data.stages[i].title+' — '+proof.label);button.append(el('span','proof-mark',proof.mark),el('span','proof-label',proof.label));button.addEventListener('click',()=>goalDialog(g.title+' · '+data.stages[i].title,[data.stages[i].definition,(g.source_available?proof.label+'. '+g.evidence:'Evidence file unavailable.'),'Next test: '+g.next,freshText(g)],g.source_url));td.append(button);tr.append(td);});body.append(tr);
 }table.append(body);if(selectedGoal){const id=selectedGoal;selectedGoal=null;selectGoal(id);}
 const jobs=$('goal-investigations'),opened=new Set([...jobs.querySelectorAll('details[open]')].map(d=>d.dataset.job));jobs.replaceChildren();
 for(const [index,j]of data.investigations.entries()){const d=el('details','goal-job');d.dataset.job=j.id;d.open=opened.has(j.id);const summary=el('summary'),title=el('span');title.append(el('span','job-order',String(index+1)),el('strong','',j.title));const status=j.id==='planner'&&a&&['experiment','qualification'].includes(a.kind)?({running:'Copied-model test running',starting:'Restoring private worlds',stale:'Test update overdue',waiting:'Waiting for test report',complete:a.kind==='qualification'?(a.passed?'Qualification passed':'Checks need repair'):'Comparison complete'}[a.state]||j.status):j.status;summary.append(title,el('span','job-status',status));if(j.freshness?.out_of_date)summary.append(el('span','job-status stale','Out of date'));d.append(summary);
  if(j.source_available){d.append(el('p','',j.finding));for(const [label,txt]of [['Action now',j.next],['Complete when',j.completion]]){const p=el('p');p.append(el('strong','',label+': '),document.createTextNode(txt));d.append(p);}const dated=cardDate(j);if(dated)d.append(dated);goalSource(d,j.source_url);}else d.append(el('p','','Evidence file unavailable; this finding needs a source check.'));jobs.append(d);}
 // open parts stay open across the live refreshes (the Now card now changes at every training report)
 const events=$('goal-events'),openEvents=new Set([...events.querySelectorAll('details[open]')].map(d=>d.dataset.event));events.replaceChildren();for(const e of data.events){const article=el('article','goal-event');article.append(el('span','event-phase',e.phase),el('h3','',e.title));if(e.step!==null&&e.step!==undefined)article.append(el('span','small muted','Training step '+stepText(e.step)));const dated=cardDate(e,true);if(dated)article.append(dated);
  if(e.source_available){article.append(el('p','',e.impact));const d=el('details');d.dataset.event=e.title+'|'+e.step;d.open=openEvents.has(d.dataset.event);d.append(el('summary','','Impact on past training'),el('p','',e.retained));if(e.fixes?.length){const ul=el('ul','repair-fixes');for(const fix of e.fixes){const li=el('li');li.append(el('strong','',fix.name+': '),document.createTextNode(fix.impact));ul.append(li);}d.append(ul);}d.append(el('p','','Next: '+e.next));goalSource(d,e.source_url);article.append(d);}else article.append(el('p','','Source unavailable.'));events.append(article);}
 renderAbilityAtlas(data);
 $('hand-program-link').hidden=!data.hand_program_url;if(data.hand_program_url)$('hand-program-link').href=data.hand_program_url;
 const map=$('upgrade-goal-map');map.replaceChildren();for(const row of state.gain_of_function?.items||[]){const p=el('p'),b=el('button','text-button',upgradeTitle(row));b.type='button';b.addEventListener('click',()=>openUpgrade(row.item));p.append(b,document.createTextNode(' → '+data.capabilities.filter(c=>c.upgrades.includes(row.item)).map(c=>c.title).join(' · ')));map.append(p);}
}

function showPage(name,update=true){
 if(!tabs.some(t=>t.dataset.page===name))name='now';
 for(const b of tabs){const active=b.dataset.page===name;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;$(b.dataset.page).hidden=!active;}
 if(name!=='video'){$('outside').pause();$('eyes').pause();}
 // deferred: showPage first runs before the rest of this script has defined the lineage state
 if(name==='history')setTimeout(()=>loadLineage(),0);
 if(name==='observer')setTimeout(()=>loadObserver(true),0);
 if(name==='tests')setTimeout(()=>loadTests(),0);
 if(name==='terms')setTimeout(()=>{if(typeof loadTermsTab==='function')loadTermsTab();},0);
 if(name==='live'||name==='viewer')setTimeout(()=>liveTick(),0);
 if(update)history.replaceState(null,'','#'+name);
}
tabs.forEach((b,i)=>{b.addEventListener('click',()=>showPage(b.dataset.page));b.addEventListener('keydown',e=>{let n;if(e.key==='ArrowRight')n=(i+1)%tabs.length;if(e.key==='ArrowLeft')n=(i+tabs.length-1)%tabs.length;if(e.key==='Home')n=0;if(e.key==='End')n=tabs.length-1;if(n!==undefined){e.preventDefault();tabs[n].focus();showPage(tabs[n].dataset.page);}});});
document.querySelectorAll('[data-open-page]').forEach(b=>b.addEventListener('click',()=>{showPage(b.dataset.openPage);tabs.find(t=>t.dataset.page===b.dataset.openPage).focus();window.scrollTo(0,0);}));
showPage(location.hash.slice(1),false);addEventListener('hashchange',()=>showPage(location.hash.slice(1),false));
// an address that names a tab (#now, #tests) is the tab's id, so the browser scrolled the page past the header and its
// machine mode band when the page loaded: a tab opened by its address starts at the top of the page
addEventListener('load',()=>{if(tabs.some(t=>t.dataset.page===location.hash.slice(1)))window.scrollTo(0,0);});

function recordStatus(r){
 if(!r)return ['No test on the running model','No reviewed test is attached for this ability',''];
 if(!r.valid)return ['Needs review','Source changed or unavailable','amber'];
 // evidence from another lineage (the 2026-09-10/11 separate predictors tested H69) says so first
 if(r.lineage!=='ancestor_checkpoint')return ['Earlier lineage only',(r.kind==='offline_component'?'A separate predictor, tested on ':'Tested on ')+'an earlier lineage, not on the running model','amber'];
 if(r.validation==='automated_validated_observation')return ['Behavior measured','Automatic observation; no ability pass','blue'];
 if(r.kind!=='offline_component')return [r.outcome==='demonstrated'?'Demonstrated in a test':'Test result recorded','Current retention untested','blue'];
 if(r.outcome==='no_added_benefit')return ['No added benefit shown','Separate choice test','amber'];
 return ['Tested separately','Not a live-model gain','blue'];
}
function placeDetail(){const button=$('ability-rows').querySelector('[data-ability="'+selectedAbility+'"]');if(compact.matches&&button)button.after($('ability-detail'));else document.querySelector('.work').append($('ability-detail'));}
compact.addEventListener('change',placeDetail);
function sourceLinks(r){const p=el('div','source-links');for(const s of r?.sources||[]){const a=el('a','',s.label+' ↗');a.href=s.url;p.append(a);}return p;}
function selectAbility(id){
 if(!state)return;const a=state.abilities.find(x=>x.id===id);if(!a)return;selectedAbility=id;
 const r=a.evidence,[label]=recordStatus(r),panel=$('ability-detail');
 for(const b of $('ability-rows').querySelectorAll('button.ability'))b.setAttribute('aria-pressed',String(b.dataset.ability===id));
 panel.replaceChildren(el('div','eyebrow',label),el('h3','',a.title));
 if(r)panel.append(el('p','date-line','Evidence dated '+r.date+' · '+(r.kind==='offline_component'?'separate experiment':'tested checkpoint')));
 const found=!r?'No reviewed behavior result is attached for this ability.':!r.valid?'The saved source no longer matches the reviewed result. Its claim is withheld until checked.':r.summary;
 const missing=!r?'A controlled test of this ability on an identified model.':!r.valid?r.review_issue:r.missing;
 const dl=el('dl');for(const [title,text] of [['What it would look like',a.looks],['What the test showed',found],['What is missing',missing],['Next proof',a.next]])dl.append(el('dt','',title),el('dd','',text));panel.append(dl);
 if(r){const d=el('details');d.append(el('summary','','Evidence and method'));if(r.valid)d.append(el('p','',r.method));d.append(el('p','',r.subject),sourceLinks(r));panel.append(d);}
 placeDetail();
}
function renderAbilities(){
 // Move the compact detail out before replacing the buttons.
 document.querySelector('.work').append($('ability-detail'));$('ability-rows').replaceChildren();
 for(const a of state.abilities){const b=el('button','ability');b.type='button';b.dataset.ability=a.id;b.setAttribute('aria-pressed','false');b.setAttribute('aria-controls','ability-detail');
  const name=el('span');name.append(el('strong','',a.title),el('span','description',a.sub));const [label,scope,tone]=recordStatus(a.evidence),status=el('span','state '+tone);status.append(el('b','',label),document.createTextNode(scope));b.append(name,status);b.addEventListener('click',()=>selectAbility(a.id));$('ability-rows').append(b);}
 selectAbility(selectedAbility);
 const attached=state.abilities.map(a=>a.evidence).filter(Boolean),here=attached.filter(r=>r.lineage==='ancestor_checkpoint').length;
 $('ability-foot').textContent='Each result belongs to the model that took the test. '+(here?here+' of the '+attached.length+' attached results tested the running model’s ancestry; ':'None of the '+attached.length+' attached results tested the running model or its ancestry; ')+'a separate predictor does not complete the running model’s ability.';
}
// The headline is generated from the ten gain-of-function items and the newest record
// (the hand-written 2026-09-13 headline of work.json described an earlier lineage).
const RESTART_NOTE='This part needs the dashboard server restarted to load its new code; the page fills it in after the restart.';
function renderHeadline(){
 const h=state.headline;
 if(state.now_summary){
  const n=state.now_summary;
   $('headline').textContent=n.title;$('headline-detail').textContent=n.evidence;
   $('prediction-comparison').textContent=n.comparison||'No command-aware prediction comparison is recorded.';
   $('prediction-scale').textContent=n.scale||'';$('prediction-sample').textContent=n.sample||'';$('progress-meaning').textContent=n.significance;
   $('current-controller').textContent=n.controller;$('body-worlds').textContent=n.worlds;
   $('synergy-reason').hidden=n.synergy!=='Learning from movement; not yet directing it';
   renderTrainingBrief();
   const passed=$('passed-measurements');passed.replaceChildren();
   for(const result of n.passed||[]){const b=el('button','text-button',result.name+' — view comparison');b.type='button';b.addEventListener('click',()=>{
    $('gof-overview').open=true;const card=$('gof-items').querySelector('[data-item="'+result.item+'"]');if(card){card.querySelector('.gof-reading').open=true;card.scrollIntoView({block:'start',behavior:'smooth'});card.tabIndex=-1;card.focus({preventScroll:true});}
   });passed.append(b);}
  $('synergy-state').textContent=n.synergy;$('latest-finding').textContent=n.synergy_detail;
  $('next-proof').textContent=n.next;
  $('run-changes').textContent=state.what_changed||'No launch change record.';
  // a schedule is not a measurement: the scheduled-test sentence stays in its own place
  $('measurement-summary').textContent=[n.comparison,n.significance,state.training_brief?.test_scheduled?null:state.training_brief?.test_status].filter(Boolean).join(' ');
  return;
 }
 if(h){
  $('headline').textContent=h.title;$('headline-detail').textContent=h.detail;
  $('latest-finding').textContent=state.what_changed||'The newest segment’s launch record is unavailable.';
  $('next-proof').textContent=state.what_next||'';
  return;
 }
 $('headline').textContent='The gain-of-function summary is not available from this server yet.';
 $('headline-detail').textContent=RESTART_NOTE;
 $('latest-finding').textContent=RESTART_NOTE;$('next-proof').textContent=RESTART_NOTE;
}
function renderTrainingBrief(){
 const b=state.training_brief;if(!b)return;
 for(const [id,key] of [['training-experience','experience'],['training-elapsed','progress'],['test-schedule','test_status'],['test-duration','test_duration'],['training-finish','finish'],['practice-estimate','practice_estimate'],['ability-timing','ability_timing']])$(id).textContent=b[key]||'';
 $('test-schedule').classList.toggle('schedule-gap',!b.test_scheduled);
 const panel=$('training-timeline');const opened=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.dataset.event));panel.replaceChildren();
 for(const [index,e] of (b.events||[]).entries()){
  const li=el('li'),heading=el('div','timeline-heading');heading.append(el('strong','',e.title));
  const due=Number.isFinite(e.due)?clockAt(e.due):null;
  const clock=el('span','timeline-time',e.status+(due?' - about '+due:''));clock.dataset.event=String(index);heading.append(clock);
  li.append(heading,el('p','',e.detail));const d=el('details');d.dataset.event=String(index);d.open=opened.has(String(index));d.append(el('summary','','How this estimate is calculated'),el('p','small',e.calculation));li.append(d);panel.append(li);
 }
}
function updateTimelineClocks(now){
 const t=state.training||{};
 const fresh=!apiError&&now-lastApiAt<=75&&state.verdict?.state==='running'&&Number.isFinite(t.record_timestamp)&&now-t.record_timestamp<600&&Number.isFinite(t.supervisor_timestamp)&&now-t.supervisor_timestamp<180;
 for(const node of document.querySelectorAll('.timeline-time')){
  const e=state.training_brief?.events?.[Number(node.dataset.event)];if(!e)continue;
  if(!fresh||!Number.isFinite(e.due)){node.textContent='Estimate unavailable while training is paused or unconfirmed';continue;}
  const stamp=clockAt(e.due);
  node.textContent=e.due<now?'Overdue - completion not recorded (estimated '+stamp+')':'About '+stamp+' (in '+age(e.due-now)+')';
 }
 $('practice-estimate').hidden=!fresh;
}
function renderVerdict(now){
 const v=state.verdict,t=state.training||{},facts=$('verdict-facts');
 let text,kind;
 if(v){text=v.sentence;kind=v.state;facts.replaceChildren(...(v.facts||[]).map(f=>el('li','',f)));}
 else{
  // older server: the supervisor's own words, without a verdict
  kind=t.pause_reason?'paused':'unconfirmed';
  text=t.pause_reason?'Training is paused: the supervisor reports "'+t.pause_reason+'".':'Training activity is not confirmed: the supervisor reports "'+(t.supervisor_status||'unknown')+'".';
  facts.replaceChildren();
 }
  if(kind==='running'&&(apiError||now-lastApiAt>75||!Number.isFinite(t.supervisor_timestamp)||now-t.supervisor_timestamp>=180||!Number.isFinite(t.record_timestamp)||now-t.record_timestamp>=600)){
   kind='unconfirmed';text='Training status is unconfirmed: current records are unavailable or overdue.';facts.replaceChildren();
  }
  if(Number.isFinite(t.record_timestamp)&&kind!=='running')text+=' Last training record: '+age(now-t.record_timestamp)+' ago.';
 if(kind==='running')text=t.view==='experiment'
  ?'Experiment run '+t.run_name+' is training, outside the production supervisor · model step '+stepText(t.step)+(Number.isFinite(t.record_timestamp)?' · newest record '+clockAt(t.record_timestamp)+', '+age(now-t.record_timestamp)+' ago':'')
  :'Training is running · step '+stepText(t.step)+(Number.isFinite(t.record_timestamp)?' · recorded '+age(now-t.record_timestamp)+' ago':'');
 if($('training-reason'))$('training-reason').textContent=v?.sentence||text;
 if(kind==='paused')text='Training paused'+(Number.isFinite(v?.final_checkpoint_step)?' · saved at step '+stepText(v.final_checkpoint_step):'')+(Number.isFinite(t.record_timestamp)?' · last report '+age(now-t.record_timestamp)+' ago':'');
 $('training-now').textContent=text;
 $('verdict').className='verdict-line '+(kind==='running'?'running':kind==='paused'?'paused':'attention');
}
// Every sentence of an item card is built by the server (lineage_series.gain_of_function, one
// `lines` list per item, DESIGN.md section 11); the page only lays them out. The main lines stay
// visible; what is logged beside the item, the field read and how its goal line would be measured
// sit in a collapsed part; the probe's own words in another.
const GOF_ASIDE=new Set(['related','how','source']);
function shortReading(text){
 const m=String(text).match(/^([+-]?\d*\.?\d+)%/);
 if(m&&Number(m[1])!==0&&Math.abs(Number(m[1]))<.001)return 'Less than 0.001% '+(Number(m[1])<0?'below':'above')+' zero';
 return text;
}
function gofItem(row){
 const a=el('article','gof-item '+(row.state||'unmeasured'));
 const head=el('div','gof-head');head.append(el('span','gof-number',String(row.item)),el('h3','',row.name));a.append(head);
 a.append(el('p','gof-question',row.question));
 if(!Array.isArray(row.lines)){a.append(el('p','small muted',RESTART_NOTE));return a;}
  a.append(el('strong','gof-value'+(row.state==='measured'?'':' unmeasured'),shortReading(row.big||'Unmeasured')));
 if(row.answer)a.append(el('p','gof-answer',row.answer));
 if(row.test_scope)a.append(el('p','test-scope',row.test_scope));
  const reading=el('details','gof-reading');reading.append(el('summary','','Measurements and comparison'));
  if(shortReading(row.big)!==row.big)reading.append(el('p','small','Recorded value: '+row.big+'. The scale and comparison are defined below.'));
 let own=null;const aside=[];
 for(const ln of row.lines||[]){
  if(ln.kind==='details'){if(!own){own=el('details','gof-own');own.append(el('summary','','The probe’s own words'));}own.append(el('p','small',ln.text));continue;}
  if(GOF_ASIDE.has(ln.kind)){aside.push(ln);continue;}
  reading.append(el('p','gof-line '+ln.kind,ln.text));
 }
 if(own)reading.append(own);
 if(aside.length){const d=el('details','gof-aside');d.append(el('summary','','Method and source'));for(const ln of aside)d.append(el('p','gof-line '+ln.kind,ln.text));reading.append(d);}
 a.append(reading);
 return a;
}
function renderGof(){
 const g=state.gain_of_function,panel=$('gof-items');
 if(!g){panel.replaceChildren(el('p','empty-history',RESTART_NOTE));$('gof-status').textContent='Not available from this server yet.';return;}
 const open=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.closest('[data-item]')?.dataset.item+':'+d.className));
 panel.replaceChildren(...g.items.map(r=>{const a=gofItem({...r,name:upgradeTitle(r)});a.dataset.item=r.item;return a;}));
 for(const d of panel.querySelectorAll('details'))if(open.has(d.closest('[data-item]').dataset.item+':'+d.className))d.open=true;
 $('gof-status').textContent=g.measured+' of '+g.total+' items measured directly'+(g.stand_in?', '+g.stand_in+' by a stand-in only':'')+' on the running model’s lineage';
}
function renderLiveness(){
 const v=state.liveness,panel=$('liveness-lines');if(!panel)return;
 const responseName=issue=>issue.source==='efference'&&issue.name==='Vision_in'?'Gaze copy':issue.name;
 const alert=$('model-alert');if(alert){alert.hidden=v?.status!=='attention';const issue=v?.issues?.[0],scope=state.training?.pause_reason?'Last production report: ':'';alert.textContent=v?.status==='attention'?scope+(v.below===1&&issue?responseName(issue)+': '+(issue.reading??issue.responding)+' of 100 units responding; warning line is 50'+(issue.definition?'. '+String(issue.definition).replace(/\.$/,''):'')+' — inspect readings':v.below+' model parts have under half their units responding'+((v.issues||[]).every(i=>String(i.verdict||'').startsWith('No action needed'))?'; each was measured as costing no accuracy':'; '+(v.issues||[]).filter(i=>!String(i.verdict||'').startsWith('No action needed')).length+' needs a new check')):'';alert.onclick=()=>{$('current-detail').open=true;$('liveness-head').scrollIntoView({block:'center',behavior:'smooth'});};}
 if(!v){panel.replaceChildren(el('p','empty-history',RESTART_NOTE));$('liveness-title').textContent='';return;}
 const opened=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.className));
 $('liveness-title').textContent=v.summary||v.title;
 $('liveness-title').closest('section').dataset.health=v.status||'unmeasured';
 const rows=[];
 for(const issue of v.issues||[]){
   const p=el('p','health-issue');p.append(el('strong','',responseName(issue)+': '),document.createTextNode((issue.reading??issue.responding)+' of 100 units responding; the warning line is 50.'));
  rows.push(p);
  if(issue.definition)rows.push(el('p','small',issue.definition));
  if(issue.verdict){const verdict=el('p','small',issue.verdict);if(issue.verdict_source){verdict.append(' ');const a=el('a','','Measurement');a.href=issue.verdict_source;verdict.append(a);}rows.push(verdict);}
 }
 if(v.repair_summary)rows.push(el('p','small',v.repair_summary));
 if(v.explanation)rows.push(el('p','small muted',v.explanation));
 if(v.glossary?.length){const g=el('details','health-glossary');g.append(el('summary','','Definitions of every part this card watches'));for(const line of v.glossary)g.append(el('p','small muted',line));rows.push(g);}
 if(v.source_text)rows.push(el('p','small muted',v.source_text));
 const current=el('details','health-current');current.append(el('summary','','Warning evidence and response checks'));
 if(v.scope_explanation)current.append(el('p','small',v.scope_explanation));
 const pairedReadings=(v.readings||[]).filter(r=>r.name.includes(' — '));
 if(pairedReadings.length){
  const table=el('table','results-table'),head=el('tr');head.append(el('th','','Measured activity'),el('th','','Responding units'));table.append(head);
  for(const reading of pairedReadings){const row=el('tr');row.append(el('td','',reading.name),el('td',reading.below?'health-issue':'',reading.responding+' of 100'+(reading.below?' — below warning line':'')));table.append(row);}current.append(table);
 }
 const known=el('details','health-history');known.append(el('summary','','Earlier findings on checkpoint 344,400 (before the 2026-09-23 upgrade)'));
  for(const ln of v.lines||[]){if(v.attention?.length&&ln.kind==='measure'&&ln.text.includes(': attention spread '))continue;const p=el('p','gof-line '+ln.kind,ln.text.charAt(0).toUpperCase()+ln.text.slice(1));(ln.kind==='known'||ln.kind==='related'?known:current).append(p);}
  if(v.attention?.length){const table=el('table','results-table'),head=el('tr');head.append(el('th','','Component'),el('th','','Attention spread'));table.append(head);
   current.append(el('p','small','Attention spread: 0% means one input; 100% means an even spread across inputs.'));
   for(const reading of v.attention){const row=el('tr');row.append(el('td','',reading.name),el('td','',reading.spread));table.append(row);}current.append(table);
  }
 if(current.children.length>1)rows.push(current);
 // each earlier line links its own dated file (the card used to link the sweep beside the readability line)
 if(known.children.length>1){const sources=v.history_sources||(v.source?[{label:'Open the earlier measurement',url:v.source}]:[]);for(const s of sources){const a=el('a','small source-link',s.label+' ↗');a.href=s.url;known.append(a);}rows.push(known);}
 panel.replaceChildren(...rows);
 for(const d of panel.querySelectorAll('details'))d.open=opened.has(d.className);
}
function historyItem(r){const article=el('article','event'),date=el('time','',r.date);date.dateTime=r.date;const body=el('div');body.append(el('h3','',r.title),el('p','',r.valid?r.summary:'Source unavailable or changed; this claim needs review.'));
 const detail=el('details');detail.append(el('summary','','Open the evidence'));if(r.valid)detail.append(el('p','',r.method));detail.append(sourceLinks(r));body.append(detail);
 const badge=el('div');badge.append(el('span','pill '+(!r.valid||['comparison_failed','no_added_benefit'].includes(r.outcome)?'amber':'gray'),!r.valid?'Needs review':r.kind==='implementation'?'System change':r.kind==='offline_component'?'Separate experiment':'Current retention untested'));
 article.append(date,body,badge);return article;}
// the reviewed-test summary also names the scheduled comparisons (the server's sentence), which
// have run on this lineage's saved checkpoints since step 380,550 and are shown on the Now tab
function historySummary(){
 const model=(state.records||[]).filter(r=>r.kind!=='implementation'&&r.lineage==='ancestor_checkpoint'),scheduled=state.ability_results?.summary;
 $('live-history-summary').textContent=model.some(r=>r.valid)?model.filter(r=>r.valid).length+' dated tests of this model’s ancestry are attached. The recorded limits are real; no new retained skill is established by these observations.':'No reviewed test from the 2026-09-13 registry has run on the running model’s ancestry.'+(scheduled?'':' That is a missing measurement, not a finding of zero ability; the gauges above and the probe suite are what measure it now.');
 const also=$('scheduled-history-summary');if(also){also.textContent=scheduled||'';also.hidden=!scheduled;}
}
function renderHistory(){
 // The running model's ancestry keeps its own list; everything else (all 7 records of
 // 2026-09-10 to 09-13 tested H69-H80 checkpoints) sits in the collapsed Earlier lineages.
 const here=r=>r.lineage==='ancestor_checkpoint';
 const model=state.records.filter(r=>r.kind!=='implementation'&&here(r));
 const project=state.records.filter(r=>r.kind==='offline_component'&&!here(r));
 const other=state.records.filter(r=>!['offline_component','implementation'].includes(r.kind)&&!here(r));
 const system=state.records.filter(r=>r.kind==='implementation');
 const earlier=project.length+other.length+system.length,jobs=(state.work?.jobs||[]).length;
 historySummary();
 $('project-history-summary').textContent=earlier+' reviewed records and '+jobs+' experiment jobs belong to earlier lineages.';
 $('earlier-summary').textContent='Earlier lineages (H69-H80): '+earlier+' reviewed records and '+jobs+' experiment jobs';
 for(const [id,records,empty] of [['model-history',model,'No reviewed behavior tests have been linked to this model’s ancestry. This is a missing measurement, not a finding of zero ability.'],['project-history',project,'No reviewed experiments are attached.'],['other-history',other,'No behavior tests from other branches are attached.'],['implementation-history',system,'No reviewed system changes are attached.']]){$(id).replaceChildren();if(!records.length)$(id).append(el('div','empty-history',empty));else for(const r of records)$(id).append(historyItem(r));}
 const r=state.records.find(x=>x.id==='wellness-choice'&&x.valid),panel=$('choice-comparison');panel.hidden=!r;if(!r)return;
 panel.replaceChildren(el('h3','','Did the separate forecasts choose better?'),el('p','small muted','Recorded comparison · '+r.date+' · '+r.worlds+' test worlds · the tested planner is an earlier checkpoint, not an evaluation of today’s changing weights.'));
 const table=el('table','results-table'),thead=el('thead'),tr=el('tr');for(const title of ['Comparison','Average benefit (share of the 0-to-3 scale)','Reported interval'])tr.append(el('th','',title));thead.append(tr);table.append(thead);const tbody=el('tbody');
 // a benefit on the 0-to-3 wellness scale, as a share of that scale (DESIGN.md section 11)
 const signed=v=>(v>=0?'+':'')+fx(100*v/3)+'%';
 for(const [label,key] of [['Forecasts versus tested planner','benefit_vs_planner'],['Forecasts versus holding still','benefit_vs_hold']]){const c=r.comparison[key],row=el('tr');for(const value of [label,signed(c.mean),c.ci95.map(signed).join(' to ')])row.append(el('td','',value));tbody.append(row);}table.append(tbody);panel.append(table);
 panel.append(el('p','result-note','Bodily wellness totals comfort, muscle capacity and posture: 0 is the lowest combined score, 3 the highest; a benefit is shown as a share of that 0-to-3 range. A benefit of 0% means no average difference; positive favors the forecasts. The planner comparison includes both loss and benefit, so improvement is unproven. These reported intervals do not prove equivalence or safety.'),sourceLinks(r));
}

const outside=$('outside'),eyes=$('eyes'),play=$('play'),seek=$('seek');
// Both files carry the same scene audio. Only the outside player supplies sound.
const sound=$('sound'),volume=$('volume');
let lastAudibleVolume=1;
function updateAudio(){
 eyes.muted=true;
 const quiet=outside.muted||outside.volume===0;
 sound.textContent=quiet?'Sound off':'Sound on';
 sound.setAttribute('aria-label',quiet?'Unmute recording':'Mute recording');
 sound.setAttribute('aria-pressed',String(quiet));
 volume.value=Math.round(outside.volume*100);
 $('volume-level').textContent=quiet?'Muted':volume.value+'%';
}
let workFilter='focus',workSignature='';
const workLabels={running:'Running',starting:'Starting',stopped:'Process ended without a status record',status_stale:'No status update for over 5 minutes',failed:'Run failed',completed_unreviewed:'Finished · not reviewed',result_missing:'Finished · report file missing',record_error:'Record unreadable',insufficient:'Finished · too little improvement',diagnosed:'Finished · fault measured',supported:'Finished · supported in this test',measured:'Finished · behavior recorded',not_started:'Not started',blocked:'Waiting for another job',files_missing:'Result files missing',no_state_recorded:'No state recorded'};
Object.assign(workLabels,{queued:'Queued for the developmental queue runner',repair_required:'Failed every allowed attempt',implemented:'Implemented · behavior not qualified',cancelled:'Stopped · instrument replaced'});
workLabels.waiting_for_resources='Waiting for memory or card headroom';
workLabels.building='Building and checking';
const workGroup=j=>j.completed_at||['cancelled','implemented','completed_unreviewed','result_missing','insufficient','diagnosed','supported','files_missing'].includes(j.state)?'complete':['not_started','blocked','queued','no_state_recorded'].includes(j.state)?'next':'active';
function experimentCard(j){
 const article=el('article','experiment-card '+workGroup(j)),header=el('div','experiment-card-header');article.dataset.job=j.id;
 header.append(el('span','pill '+(['running','supported'].includes(j.state)?'':'amber'),j.review_issue?'Result changed after review':workLabels[j.state||'no_state_recorded']||('State "'+j.state+'"')),el('span','small muted',j.owner||'Project work'));article.append(header,el('h3','',j.title),el('p','experiment-question',j.question));
 if(j.result)article.append(el('p','experiment-result',j.result));
 if(j.measurement?.text)article.append(el('p','small muted',j.measurement.text));
 if(j.observations){const table=el('table','results-table'),body=el('tbody');for(const o of j.observations){const row=el('tr');row.append(el('th','',o.label),el('td','',o.display||(100*o.value).toFixed(0)+'%'));body.append(row);}table.append(body);article.append(table,el('p','small muted',j.observation_scale));}
 const p=j.progress;
 if(p){
  const label=el('div','progress-label');label.append(el('strong','',p.display||Number(p.done).toLocaleString()+' / '+Number(p.total).toLocaleString()),el('span','',p.display?'Learning phase':p.label));article.append(label);
  const bar=el('progress');bar.max=p.total;bar.value=p.done;bar.setAttribute('aria-label',p.label);article.append(bar,el('p','small muted',p.detail));
  if(p.phases){const phases=el('ol','experiment-phases');for(const phase of p.phases)phases.append(el('li',phase.state,phase.name+' · '+({complete:'done',running:'underway',waiting:'not begun'}[phase.state]||phase.state)));article.append(phases);}
 }
 if(j.completed_at)article.append(el('p','experiment-when','Finished '+dated(j.completed_at)+'.'));
 else if(j.started_at)article.append(el('p','small muted','Started '+dated(j.started_at)+'. Latest record '+(j.updated_at?dated(j.updated_at):'not yet written')+'.'));
 else if(j.when)article.append(el('p','experiment-when',j.when));
 if(j.failure||j.review_issue)article.append(el('p','experiment-error',j.failure||j.review_issue));
 if(j.limitation)article.append(el('p','experiment-error',j.limitation));
 const decision=el('div','experiment-decision');decision.append(el('strong','',workGroup(j)==='complete'?'Decision':'What the result decides'),el('p','',j.decision));article.append(decision);
 if(j.scope){const detail=el('details');detail.append(el('summary','','What this test covers'),el('p','',j.scope));article.append(detail);}
 const links=el('div','experiment-links');for(const link of j.links||[]){const a=el('a','',link.label);a.href=link.url;links.append(a);}article.append(links);return article;
}
function renderWork(force=false){
 renderDevelopment();
 const board=state.work||{jobs:[]},signature=JSON.stringify([board,workFilter]);if(!force&&signature===workSignature)return;workSignature=signature;
 const jobs=board.jobs||[],panel=$('experiment-jobs'),opened=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.closest('[data-job]').dataset.job));panel.replaceChildren();
 if(board.queue_pause_reason)panel.append(el('p','experiment-error',board.queue_pause_reason));
 const visible=workFilter==='focus'?jobs.filter(j=>!j.superseded_by).sort((a,b)=>({active:0,next:1,complete:2}[workGroup(a)]-{active:0,next:1,complete:2}[workGroup(b)])).slice(0,4):jobs.filter(j=>workFilter==='all'||workGroup(j)===workFilter);
 for(const j of visible){const card=experimentCard(j);if(opened.has(j.id)&&card.querySelector('details'))card.querySelector('details').open=true;panel.append(card);}
 if(!visible.length)panel.append(el('p','empty-history',board.error?'Experiment records unavailable: '+board.error:'No experiments in this view.'));
 const missing=jobs.filter(j=>j.state==='files_missing').length;
 $('work-status').textContent=jobs.length+' jobs: '+jobs.filter(j=>j.state==='running').length+' running, '+missing+' with result files missing, '+jobs.filter(j=>workGroup(j)==='complete'&&!['cancelled','files_missing'].includes(j.state)).length+' with results, '+jobs.filter(j=>j.state==='cancelled').length+' retired, '+jobs.filter(j=>workGroup(j)==='next').length+' never run';
 $('work-history').replaceChildren();for(const j of jobs.filter(j=>workGroup(j)==='complete').sort((a,b)=>(b.completed_at||0)-(a.completed_at||0)))$('work-history').append(experimentCard(j));
}
let developmentSignature='';
function renderDevelopment(){
 const data=state.development||{stages:[]},signature=JSON.stringify(data);if(signature===developmentSignature)return;developmentSignature=signature;
 const panel=$('development-stages');if(!panel)return;
 if(data.definitions)$('definitions-link').href=data.definitions;
 if(data.definitions_label)$('definitions-link').textContent=data.definitions_label;
 const opened=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.dataset.abilityId));panel.replaceChildren();
 for(const stage of data.stages){
  const section=el('section','development-stage');section.append(el('h3','','Stage '+stage.number+' · '+stage.title));
  for(const ability of stage.abilities){
   const row=el('details','development-ability');row.dataset.abilityId=ability.id;row.open=opened.has(ability.id);
   const summary=el('summary');summary.append(el('strong','',ability.name),el('span','small muted',ability.status));row.append(summary);
   const dl=el('dl');for(const [label,value] of [['What the test asks',ability.question],['Why act while well?',ability.driver],['Minimum environment',ability.minimum_environment],['Infant comparison',ability.human_comparison],['What must be resolved',ability.prerequisite]])dl.append(el('dt','',label),el('dd','',value));row.append(dl);
   for(const observed of ability.observations||[]){row.append(el('p','experiment-result',observed.summary),el('p','small muted',observed.limitation),sourceLinks(observed));}
   for(const job of ability.experiments){const where=job.lineage==='ancestor_checkpoint'?'tested this model’s ancestry':job.lineage==='not_linked_to_current_model'?'an earlier lineage':job.lineage==='unknown'?'tested model not recorded':'';const p=el('p','small',job.title+' · '+(workLabels[job.state]||job.state||'no state recorded')+(where?' · '+where:'')+(job.failure?' · '+job.failure:''));for(const link of job.links||[]){const a=el('a','',link.label);a.href=link.url;p.append(document.createTextNode(' · '),a);}row.append(p);}
   section.append(row);
  }panel.append(section);
 }
}
document.querySelectorAll('[data-work-filter]').forEach(b=>b.addEventListener('click',()=>{workFilter=b.dataset.workFilter;for(const other of document.querySelectorAll('[data-work-filter]'))other.setAttribute('aria-pressed',String(other===b));if(state)renderWork(true);}));
sound.addEventListener('click',()=>{
 if(outside.muted||outside.volume===0){outside.muted=false;if(outside.volume===0)outside.volume=lastAudibleVolume;}
 else outside.muted=true;
 updateAudio();
});
volume.addEventListener('input',()=>{
 outside.volume=Number(volume.value)/100;
 outside.muted=outside.volume===0;
 if(outside.volume>0)lastAudibleVolume=outside.volume;
 updateAudio();
});
outside.addEventListener('volumechange',updateAudio);
updateAudio();
function fmt(t){return Number.isFinite(t)?Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0'):'—';}
function sync(force=false){if(eyes.readyState>0&&(force||Math.abs(eyes.currentTime-outside.currentTime)>.12))eyes.currentTime=outside.currentTime;$('clock').textContent=fmt(outside.currentTime)+' / '+fmt(outside.duration);seek.value=Number.isFinite(outside.duration)&&outside.duration>0?outside.currentTime/outside.duration*1000:0;}
// which run made a clip, as the list and the clip note name it (the server's videos_for: run name, experiment state)
function videoRun(v){return v.experiment?'experiment run '+v.run_name+(v.experiment_state==='live'?' (training now)':v.experiment_state==='ended'?' (ended)':' (state not confirmed)'):(v.previous_run?'earlier run ':'current run ')+v.run_name;}
// the two notes under the player, written again on every refresh: a run that ends while its clip is selected is named ended
function videoNotes(v){const run=videoRun(v);$('clip-note').textContent=run[0].toUpperCase()+run.slice(1)+' · recording at model step '+v.step.toLocaleString()+' · saved '+dated(v.recorded_at)+'.';$('video-note').textContent=(v.kind==='event'?'This clip was cut when a “'+(v.events?.[0]||'event')+'” event happened in the watched body’s world: an event chose it, not the run’s video schedule. ':'')+(v.control_annotated?'The on-screen label identifies scheduled stillness or model commands. Labels are for the viewer; the baby does not see them. Purpose remains untested.':'This older clip did not record who controlled the body. Scheduled stillness may be present and cannot be identified from movement alone.');}
function loadVideo(v){if(!v)return;outside.pause();eyes.pause();selectedVideo=v;outside.src=v.outside;eyes.src=v.eyes;$('recordings').value=v.id;videoNotes(v);$('new-video').hidden=state.videos[0]?.id===v.id;}
function updateVideos(){
 const videos=state.videos;const available=videos.length>0||selectedVideo!==null;$('no-video').hidden=available;$('video-body').hidden=!available;$('recordings').disabled=!available;
 const options=[...videos];if(selectedVideo&&!options.some(v=>v.id===selectedVideo.id))options.push(selectedVideo);
 $('recordings').replaceChildren();for(const v of options){const option=el('option','','Model step '+v.step.toLocaleString()+' · '+videoRun(v)+' · '+dated(v.recorded_at)+(v.kind==='event'?' · event clip ('+(v.events?.[0]||'event')+')':''));option.value=v.id;$('recordings').append(option);}
 const fresh=selectedVideo&&videos.find(x=>x.id===selectedVideo.id);if(fresh){selectedVideo=fresh;videoNotes(fresh);}
 if(!selectedVideo&&videos.length)loadVideo(videos[0]);else if(selectedVideo){$('recordings').value=selectedVideo.id;$('new-video').hidden=!videos.length||videos[0].id===selectedVideo.id;}
}
$('recordings').addEventListener('change',()=>loadVideo(state.videos.find(v=>v.id===$('recordings').value)||selectedVideo));$('new-video').addEventListener('click',()=>loadVideo(state.videos[0]));
play.addEventListener('click',async()=>{if(!outside.paused){outside.pause();eyes.pause();return;}sync(true);try{await Promise.all([outside.play(),eyes.play()]);}catch(e){outside.pause();eyes.pause();$('video-note').textContent='This recording could not play. Choose another completed pair.';}});
outside.addEventListener('play',()=>{play.textContent='Pause';play.setAttribute('aria-label','Pause both views');});outside.addEventListener('pause',()=>{eyes.pause();play.textContent='Play';play.setAttribute('aria-label','Play both views');});outside.addEventListener('ended',()=>eyes.pause());outside.addEventListener('timeupdate',()=>sync());outside.addEventListener('loadedmetadata',()=>sync());eyes.addEventListener('loadedmetadata',()=>sync(true));seek.addEventListener('input',()=>{if(Number.isFinite(outside.duration)){outside.currentTime=outside.duration*seek.value/1000;sync(true);}});
for(const v of [outside,eyes])v.addEventListener('error',()=>{$('video-note').textContent='The selected recording is unavailable. No behavioral result is inferred from the missing video.';});

function updateObservations(){
 // every sentence is built by the server (development_dashboard.observation_cards)
 const panel=$('live-observations'),opened=new Set([...panel.querySelectorAll('details[open]')].map(d=>d.dataset.observation));panel.replaceChildren();
 const cards=state.observations;
 if(!Array.isArray(cards)){panel.append(el('p','empty-history',RESTART_NOTE));return;}
 for(const c of cards){const a=el('article');a.append(el('h3','',c.title),el('strong','observation-value',c.value));if(c.notes?.length){const d=el('details');d.dataset.observation=c.title;d.open=opened.has(c.title);d.append(el('summary','','What this reading means'));for(const n of c.notes)d.append(el('p','',n));a.append(d);}panel.append(a);}
}
function updateLegacyClock(now){
 const l=state.legacy||{},parts=[];
 for(const [key,label,maxAge] of [['dashboard','Old dashboard',180],['history','Training history',2100]]){
  const ts=l[key]?.updated_at;parts.push(Number.isFinite(ts)?label+' updated '+age(now-ts)+' ago'+(now-ts>maxAge?' (overdue)':''):label+' unavailable');
 }
 if(l.updater?.status==='failed')parts.push('Updater error: '+l.updater.error);
 $('legacy-status').textContent=parts.join(' \u00b7 ');
 $('legacy-status').classList.toggle('error',l.updater?.status==='failed'||!l.dashboard||now-l.dashboard.updated_at>180||!l.history||now-l.history.updated_at>2100);
}

function updateClocks(){
 const now=Date.now()/1000,connection=$('connection');connection.classList.toggle('error',apiError||!lastApiAt||now-lastApiAt>75);
 if(!lastApiAt){connection.textContent=apiError?'Live records unavailable. Retrying…':'Connecting to current records…';return;}
 connection.textContent=(apiError?'Live update failed · last connection ':'Updated ')+age(now-lastApiAt)+' ago · refreshes every 30 seconds';
 if(apiError||now-lastApiAt>75)$('work-status').textContent='Connection unavailable. Experiment states below were last read '+dated(lastApiAt)+'.';
 const t=state.training,hasRecord=Number.isFinite(t.record_timestamp);
 let text='Training details and launch settings';
 if(hasRecord)text+=' · newest training record '+age(now-t.record_timestamp)+' old';
 $('training-summary').textContent=text;
 renderVerdict(now);
 const automatic=$('automatic-checks')?.querySelector('button');if(automatic)automatic.textContent=automaticChecksSummary(now);
 const activity=state.goal_catalog?.activity,status=$('activity-status');if(status&&activity){
  const unavailable=apiError||now-lastApiAt>75||(activity.state==='running'&&(!Number.isFinite(activity.recorded_at)||now-activity.recorded_at>=600));for(const id of ['activity-progress','activity-remaining'])if($(id))$(id).hidden=unavailable;
  for(const due of document.querySelectorAll('.activity-due'))due.hidden=unavailable;
  if(apiError||now-lastApiAt>75)status.textContent='Current progress unavailable';
  else if(activity.state==='running'&&(!Number.isFinite(activity.recorded_at)||now-activity.recorded_at>=600))status.textContent=(['training','production'].includes(activity.kind)?'Training report overdue':'Test update overdue')+' · last count '+activity.done+' of '+activity.total;
 }
 updateTimelineClocks(now);
 updateLiveFreshness(now);
 updateLegacyClock(now);
 $('observation-time').textContent=hasRecord?'Recorded '+age(now-t.record_timestamp)+' ago'+(now-t.record_timestamp>=180?' \u00b7 observations may be stale':''):'No current observation';
}
function updateTraining(){
 const t=state.training,panel=$('training-detail');panel.replaceChildren();
 const experiment=t.view==='experiment';
 if(t.run_name)panel.append(el('p','',experiment?'Experiment run: '+t.run_name+'. It trains outside the production supervisor; this page reads the copies the experiment mirror writes into '+t.run+'.':'Run: '+t.run_name));
 if(t.step!==null&&t.step!==undefined)panel.append(el('p','','Newest recorded lineage step: '+Number(t.step).toLocaleString()+'. This counts training steps, not acquired skills.'));
 if(t.record_timestamp)panel.append(el('p','','Training record written '+dated(t.record_timestamp)+'.'));
 panel.append(el('p','',experiment?'"Running" requires the owner process named in the experiment’s owner record and the mirror process to be alive at this read, the owner record to read phase "training" with its trainer process alive, the mirror’s last pass over the run’s files to be under 3 minutes old and the newest training record under 10 minutes old. It does not certify learning or model health.':'"Running" requires the supervisor’s record to be under 3 minutes old and the newest training record under 10 minutes old. It does not certify learning or model health.'));
 if(Array.isArray(t.settings)){
  const table=el('table','results-table settings-table'),body=el('tbody');
  const head=el('tr');for(const h of ['Launch flag','Value on this run','What it does'])head.append(el('th','',h));body.append(head);
  // a launch setting is shown as the operator types it (a literal, not a score: DESIGN.md section 11)
  for(const s of t.settings){const row=el('tr'),v=el('td');v.append(el('code','',s.value));row.append(el('td','',s.flag),v,el('td','',s.meaning));body.append(row);}
  table.append(body);panel.append(el('p','','Launch settings of this run: the flags that decide behaviour, then every flag the gain-of-function build adds (each defaults to the old behaviour).'),table);
 }else if(t.settings)panel.append(el('p','',RESTART_NOTE));
 if(t.source){const a=el('a','',experiment?'Open the experiment mirror’s record ↗':'Open the supervisor record ↗');a.href=t.source;panel.append(a);}
 if(state.verdict?.source){const a=el('a','','Open the supervisor control file ↗');a.href=state.verdict.source;panel.append(document.createTextNode(' · '),a);}
}
// header links: when "How the model works" was built, and each project record's last write (server times)
function renderResources(){
 const x=state.explainer,link=$('explainer-link'),built=$('explainer-age');
 if(link&&built){built.textContent=x?x.label:'';built.hidden=!x;built.classList.toggle('stale',Boolean(x&&!x.current));if(x){link.href=x.url;link.title=x.detail;built.title=x.detail;}}
 const box=$('project-records'),records=state.project_records;
 if(!box||!Array.isArray(records)||!records.length)return;
 const signature=JSON.stringify(records);if(box.dataset.signature===signature)return;box.dataset.signature=signature;box.replaceChildren();
 for(const r of records){const a=el('a','project-record');a.href=r.url;a.append(el('span','',r.label),el('span','record-age'+(r.stale?' stale':''),r.updated));box.append(a);}
}
// the Project records menu overlays the page, so a click outside it closes it
document.addEventListener('click',e=>{for(const d of document.querySelectorAll('details.resources[open]'))if(!d.contains(e.target))d.open=false;});
async function refresh(){
 const requestedRevision=liveRevision;
 try{const response=await fetch('/api/progress',{cache:'no-store',signal:AbortSignal.timeout(12000)});if(!response.ok)throw Error('Unavailable');const next=await response.json();if(next.schema!==1||!Array.isArray(next.abilities))throw Error('Unexpected data');if(state&&(liveBusy||requestedRevision!==liveRevision||state.live?.observed_at>next.live?.observed_at))next.live=state.live;state=next;lastApiAt=Date.now()/1000;apiError=false;
  const signature=JSON.stringify([state.abilities,state.records]);if(signature!==evidenceSignature){const detailOpen=$('ability-detail').querySelector('details')?.open;evidenceSignature=signature;renderHeadline();renderAbilities();if(detailOpen&&$('ability-detail').querySelector('details'))$('ability-detail').querySelector('details').open=true;renderHistory();}
  renderHeadline();renderGof();renderGoalProgramme();renderLiveness();renderWork();updateVideos();updateTraining();updateObservations();paintLive();renderResources();historySummary();$('evidence-time').textContent='Evidence is dated individually. Page data read '+dated(state.generated_at)+'.';
 }catch(e){apiError=true;}updateClocks();
 if(!$('history').hidden)loadLineage();
 loadOverview();
}

// ---------------------------------------------------------------- gauges over the lineage
// GET /api/lineage (tools/lineage_series.py): one aligned, downsampled x axis (lineage step) for
// every series, segment markers, series definitions and the chart cards. Every sentence a panel
// prints (latest reading, scale ends, goal line, why a series is unmeasured) is built by the server
// (lineage_series.panel_texts); the page draws the charts in each series' DISPLAY form (a count per
// 100, a real unit, or a percentage between two named ends: display = a * value + b), so no axis,
// tip or table cell prints a bare score decimal (DESIGN.md section 11). Colours: the dataviz
// reference palette's first three slots, validated as a set for up to three lines; text never wears
// a series colour.
const SERIES_COLORS=['#2a78d6','#eb6834','#1baf7a'];
const SVGNS='http://www.w3.org/2000/svg';
let lineage=null,lineageAt=0,lineageError='',moreData=null,moreAt=0;
const sv=(tag,attrs,text)=>{const n=document.createElementNS(SVGNS,tag);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
// fixed notation with significant digits, never an exponent: 7.3e-6 -> "0.0000073"
function fx(v,sig=2){
 if(v==null||!Number.isFinite(v))return 'unmeasured';
 const a=Math.abs(v);if(a===0)return '0';
 if(a>=10)return Math.round(v).toLocaleString();
 if(a>=1){const t=v.toFixed(1);return t.endsWith('.0')?t.slice(0,-2):t;}
 const d=Math.min(20,Math.max(0,sig-1-Math.floor(Math.log10(a))));
 let t=v.toFixed(d);if(t.includes('.'))t=t.replace(/0+$/,'').replace(/\.$/,'');return t==='-0'?'0':t;
}
const disp=(d,v)=>(v==null||!Number.isFinite(v))?null:d.display.a*v+d.display.b;
// a display value with its unit ("12%", "0.83 cm", "4.1 GB"); a count under 1 is said per 100
function dtext(d,x){
 if(x==null||!Number.isFinite(x))return 'unmeasured';
 const D=d.display;
 if(D.kind==='count'&&x!==0&&Math.abs(x)<1)return fx(100*x)+' per 100 '+(D.per_many||'windows');
 return fx(x)+(D.unit||'');
}
const shortStep=s=>s>=1e6?Math.round(s/1e4)/100+'M steps':s>=1e3?Math.round(s/1e3)+'K steps':Math.round(s)+' steps';
const shortName=d=>d.title.includes(':')?d.title.split(':')[0]:d.title;
async function loadLineage(force=false){
 const now=Date.now()/1000;
 if(!force&&lineage&&now-lineageAt<300)return;   // at most every 5 minutes; a window is 100 steps
 lineageAt=now;
 try{
  const r=await fetch('/api/lineage',{cache:'no-store',signal:AbortSignal.timeout(60000)});
  if(r.status===404){lineageError=RESTART_NOTE;renderLineage();return;}
  if(!r.ok)throw Error('the server answered '+r.status);
  const data=await r.json();if(data.schema!==1||!data.x)throw Error('unexpected data');
  lineage=data;lineageError='';
 }catch(e){lineageError='The lineage gauges could not be read ('+e.message+'); the page retries every 5 minutes while this tab is open.';}
 renderLineage();
}
function renderLineage(){
 const cards=$('gauge-cards');
 if(!lineage){cards.replaceChildren(el('p','empty-history',lineageError||'Reading the lineage…'));$('lineage-status').textContent=lineageError?'Unavailable':'Reading the lineage.';return;}
 const L=lineage,xs=L.x.step,segs=L.segments||[];
 $('lineage-status').textContent=segs.length+' segments, lineage steps '+stepText(xs[0])+' to '+stepText(xs[xs.length-1])+' · read '+dated(L.generated_at)+(lineageError?' · '+lineageError:'');
 const open=new Set([...cards.querySelectorAll('details[open]')].map(d=>d.dataset.key));
 cards.replaceChildren(...L.cards.map(c=>gaugeCard(c,L)));
 for(const d of cards.querySelectorAll('details'))if(open.has(d.dataset.key))d.open=true;
 const table=$('segment-table');table.replaceChildren();
 const head=el('tr');for(const h of ['Segment','Lineage steps','Records','Code','Launch settings in force that changed from the segment before'])head.append(el('th','',h));table.append(head);
 for(const s of segs){const row=el('tr');row.append(el('td','',s.label),el('td','',s.first_step==null?'no records':stepText(s.first_step)+' to '+stepText(s.last_step)),el('td','',String(s.records)),el('td','',s.code||s.code_commit||'not recorded'),
  el('td','',s.fresh?'fresh weights':(s.changed||[]).length?s.changed.map(c=>c.text||(c.flag+' '+c.now)).join('; '):'none'));table.append(row);}
 $('segment-summary').textContent='Segments of this lineage ('+segs.length+'): where each begins, its code and what its launch changed';
}
function gaugeCard(card,L){
 const a=el('article','gauge-card');a.id='gauge-'+card.id;a.append(el('h3','',card.title));
 const d=el('details','chart-details');d.dataset.key=card.id+':charts';d.append(el('summary','','Charts, comparisons and definitions'),el('p','small muted',card.what));
 if(card.item===5&&state?.now_summary){d.append(el('p','small',state.now_summary.synergy_detail),el('p','small','The executed-command charts describe the current controller. They do not establish that synergy control improves movement.'));}
 const scales=new Set();for(const p of card.panels)d.append(gaugePanel(p,L,card.id,scales));
 if(card.probe_item!=null)d.append(...probePanels(card.probe_item,L));
 const panels=card.panels||[],measured=panels.filter(p=>p.keys.some(k=>L.defs[k]?.latest));
 const probes=((L.probes?.items||{})[String(card.probe_item)]||[]).filter(p=>p.latest),direct=probes.filter(p=>!p.words?.stand_in),standins=probes.filter(p=>p.words?.stand_in);
 let summary=measured.length?measured.length+' of '+panels.length+' chart groups have readings.':'No continuous chart readings.';
 if(card.probe_item!=null)summary+=' '+(direct.length?direct.length+' direct movement test'+(direct.length===1?'':'s')+' recorded.':'No direct movement test recorded.')+(standins.length?' '+standins.length+' stand-in test'+(standins.length===1?'':'s')+' available.':'');
 a.append(el('p','small muted',summary),d);
 return a;
}
function gaugePanel(panel,L,cardId,scales=new Set()){
 const box=el('div','gauge-panel');box.append(el('h4','',panel.title));
 // a server still running the code before 2026-09-23 sends no display forms or sentences
 if(!panel.texts||panel.keys.some(k=>L.defs[k]&&!L.defs[k].display)){box.append(el('p','unmeasured-note',RESTART_NOTE));return box;}
 const T=panel.texts||{},keys=panel.keys.filter(k=>L.defs[k]),has=keys.filter(k=>Array.isArray(L.series[k]));
 const scale=T.scale?(scales.has(T.scale)?'Scale: same units and endpoints as above.':'Scale: '+T.scale):null;
 if(T.scale)scales.add(T.scale);
 if(!keys.length||!has.length){
  box.append(el('p','unmeasured-note',T.missing_summary||'No measurement is recorded in this training history.'));
  if(T.unmeasured){const source=el('details');source.append(el('summary','','Measurement source'),el('p','small',T.unmeasured));box.append(source);}
  if(scale)box.append(el('p','small muted',scale));
  if(T.goal)box.append(el('p','small muted',T.goal));
  return box;
 }
 if(T.latest)box.append(el('p','gauge-latest',T.latest));
 box.append(meter(keys,L));
 for(const t of [scale,T.goal,T.band,T.seam])if(t)box.append(el('p','small muted',t));
 box.append(chart(keys,L),numbersTable(keys,L,cardId+':'+keys.join(',')));
 return box;
}
function meter(keys,L){
 // where each latest value sits between the scale's two ends, with the floor and goal marked
 const d=L.defs[keys[0]];if(!d.hi)return el('span');
 const lo=disp(d,d.lo[0]),hi=disp(d,d.hi[0]),W=640,H=30,X=v=>12+Math.max(0,Math.min(1,(v-Math.min(lo,hi))/(Math.abs(hi-lo)||1)))*(W-24);
 const svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,class:'meter','aria-hidden':'true'});
 svg.append(sv('line',{x1:12,x2:W-12,y1:15,y2:15,class:'meter-track'}));
 if(d.floor)svg.append(sv('line',{x1:X(disp(d,d.floor[0])),x2:X(disp(d,d.floor[0])),y1:6,y2:24,class:'meter-floor'}));
 if(d.goal)svg.append(sv('line',{x1:X(disp(d,d.goal[0])),x2:X(disp(d,d.goal[0])),y1:4,y2:26,class:'meter-goal'}));
 keys.forEach((k,i)=>{const l=L.defs[k].latest;if(l)svg.append(sv('circle',{cx:X(disp(L.defs[k],l.value)),cy:15,r:5,fill:SERIES_COLORS[i%3],class:'dot'}));});
 const wrap=el('div','meter-wrap');wrap.append(el('span','meter-end',dtext(d,Math.min(lo,hi))),svg,el('span','meter-end',dtext(d,Math.max(lo,hi))));return wrap;
}
function linePath(values,X,Y,xs){
 let p='',pen=false;
 values.forEach((v,i)=>{if(v==null||!Number.isFinite(v)){pen=false;return;}p+=(pen?'L':'M')+X(xs[i]).toFixed(1)+','+Y(v).toFixed(1);pen=true;});
 return p;
}
function ticksFor(d,lo,hi){
 // a count axis gets whole-number ticks; every other axis carries its unit on each tick
 if(d.display.kind==='count'){const step=Math.max(1,Math.ceil((hi-lo)/4));const out=[];for(let v=Math.floor(lo);v<=Math.ceil(hi)+1e-9;v+=step)out.push(v);return out.length>1?out:[Math.floor(lo),Math.floor(lo)+1];}
 const out=[];for(let i=0;i<=4;i++)out.push(lo+(hi-lo)*i/4);return out;
}
function chart(keys,L){
 const W=640,H=210,m={l:66,r:14,t:16,b:30},xs=L.x.step,n=xs.length;
 const d0=L.defs[keys[0]];
 const T=(d,arr)=>arr.map(v=>disp(d,v));
 const lines=keys.map((k,i)=>({k,d:L.defs[k],v:T(L.defs[k],L.series[k]||[]),color:SERIES_COLORS[i%3]})).filter(x=>Array.isArray(L.series[x.k]));
 const extra=[];for(const ln of lines){for(const [kind,fs] of [['control',ln.d.floor_series],['limit',ln.d.limit_series]])if(fs&&Array.isArray(L.series[fs])&&!extra.some(c=>c.k===fs))extra.push({k:fs,kind,d:L.defs[fs],v:T(L.defs[fs],L.series[fs])});}
 const vals=[];
 for(const ln of [...lines,...extra])for(const v of ln.v)if(v!=null&&Number.isFinite(v))vals.push(v);
 const refs=[];if(d0.floor)refs.push({v:disp(d0,d0.floor[0]),cls:'floor',label:'floor '+dtext(d0,disp(d0,d0.floor[0]))});
 if(d0.goal)refs.push({v:disp(d0,d0.goal[0]),cls:'goal',label:'goal line '+dtext(d0,disp(d0,d0.goal[0]))});
 for(const r of d0.refs||[])refs.push({v:disp(d0,r[0]),cls:'ref-line',label:r[1]+' '+dtext(d0,disp(d0,r[0]))});
 for(const r of refs)vals.push(r.v);
 const band=d0.band?[disp(d0,d0.band[0]),disp(d0,d0.band[1])]:null;if(band)vals.push(...band);
 let lo=Math.min(...vals),hi=Math.max(...vals);if(!(hi-lo>1e-12)){const p=Math.abs(hi)*0.05||0.5;lo-=p;hi+=p;}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;
 const x0=xs[0],x1=Math.max(xs[n-1],x0+1),X=s=>m.l+(s-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>m.t+(hi-v)/(hi-lo)*(H-m.t-m.b);
 const svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,class:'gauge-svg',role:'img','aria-label':lines.map(l=>l.d.title).join('; ')+' over the lineage'});
 for(const v of ticksFor(d0,lo,hi)){if(v<lo-1e-9||v>hi+1e-9)continue;const y=Y(v);svg.append(sv('line',{x1:m.l,x2:W-m.r,y1:y,y2:y,class:'grid'}),sv('text',{x:m.l-6,y:y+4,class:'tick','text-anchor':'end'},dtext(d0,v)));}
 for(let i=0;i<=4;i++){const s=x0+(x1-x0)*i/4;svg.append(sv('text',{x:X(s),y:H-10,class:'tick','text-anchor':i===0?'start':i===4?'end':'middle'},shortStep(s)));}
 if(band){const y1=Y(Math.max(...band)),y2=Y(Math.min(...band));const r=sv('rect',{x:m.l,y:y1,width:W-m.l-m.r,height:Math.max(1,y2-y1),class:'band'});r.append(sv('title',{},'shaded: '+d0.band[2]));svg.append(r);}
 for(const seg of (L.segments||[]).slice(1))if(seg.first_step!=null){const x=X(seg.first_step);const g=sv('line',{x1:x,x2:x,y1:m.t,y2:H-m.b,class:'seg'});g.append(sv('title',{},'Segment '+seg.label+' begins at lineage step '+stepText(seg.first_step)));svg.append(g);}
 // a gauge whose definition changed (lineage_series seam_text): marked where the first segment running the change begins
 const seam=d0.seam_at;if(seam&&seam.step!=null){const x=X(seam.step);const g=sv('line',{x1:x,x2:x,y1:m.t,y2:H-m.b,class:'seam'});g.append(sv('title',{},'The gauge changed here: segment '+seam.label+' is the first to run the new gauge; readings on the two sides are different measurements'));svg.append(g,sv('text',{x:x+3,y:m.t+10,class:'ref'},'gauge changed'));}
 refs.forEach((r,i)=>{const y=Y(r.v);svg.append(sv('line',{x1:m.l,x2:W-m.r,y1:y,y2:y,class:r.cls}),sv('text',{x:r.cls==='floor'?m.l+4:W-m.r-2,y:y-4-(r.cls==='ref-line'?0:0),class:'ref','text-anchor':r.cls==='floor'?'start':'end'},r.label));});
 for(const c of extra)svg.append(sv('path',{d:linePath(c.v,X,Y,xs),class:c.kind==='limit'?'limit-line':'control-line'}));
 for(const ln of lines){svg.append(sv('path',{d:linePath(ln.v,X,Y,xs),class:'series-line',stroke:ln.color}));
  let i=n-1;while(i>=0&&ln.v[i]==null)i--;if(i>=0)svg.append(sv('circle',{cx:X(xs[i]),cy:Y(ln.v[i]),r:4,fill:ln.color,class:'dot'}));}
 const cross=sv('line',{x1:0,x2:0,y1:m.t,y2:H-m.b,class:'cross',visibility:'hidden'});svg.append(cross);
 const hit=sv('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'transparent'});svg.append(hit);
 const wrap=el('div','chart-wrap');wrap.tabIndex=0;wrap.setAttribute('aria-label','Chart; use the arrow keys to read each point');
 const tip=el('div','tip');tip.hidden=true;wrap.append(svg,tip);
 if(lines.length+extra.length>1){const legend=el('div','legend-row');
  lines.forEach(ln=>{const k=el('span','key');k.append(el('i','line-key'),document.createTextNode(shortName(ln.d)));k.firstChild.style.background=ln.color;legend.append(k);});
  extra.forEach(c=>{const k=el('span','key');k.append(el('i','line-key '+c.kind),document.createTextNode((c.kind==='limit'?'limit: ':'control: ')+c.d.title));legend.append(k);});
  wrap.append(legend);}
 let at=n-1;
 const show=i=>{at=Math.max(0,Math.min(n-1,i));const x=X(xs[at]);cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
  const seg=(L.segments||[]).find(s=>s.index===L.x.segment[at]);
  const range=L.x.start_step?'Average over steps '+stepText(L.x.start_step[at])+' to '+stepText(xs[at]):'Lineage step '+stepText(xs[at]);
  const stamp=Number.isFinite(L.x.time?.[at])?' · '+dated(L.x.time[at]):'';
  tip.replaceChildren(el('div','tip-head',range+stamp+(seg?' · segment '+seg.label:'')));
  for(const ln of [...lines,...extra]){const row=el('div','tip-row');const key=el('i','line-key'+(ln.color?'':' '+ln.kind));if(ln.color)key.style.background=ln.color;const count=L.x.counts?.[ln.k]?.[at];row.append(key,el('strong','',dtext(ln.d,ln.v[at])),el('span','',(ln.color?shortName(ln.d):ln.kind)+(count!=null?' ('+count+' report windows)':'')));tip.append(row);}
  tip.hidden=false;tip.style.left=Math.min(70,Math.max(0,100*x/W-10))+'%';};
 hit.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect(),sx=(e.clientX-r.left)/r.width*W;let best=0,bd=Infinity;for(let i=0;i<n;i++){const dd=Math.abs(X(xs[i])-sx);if(dd<bd){bd=dd;best=i;}}show(best);});
 hit.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');tip.hidden=true;});
 wrap.addEventListener('focus',()=>show(at));wrap.addEventListener('blur',()=>{cross.setAttribute('visibility','hidden');tip.hidden=true;});
 wrap.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();show(at-1);}if(e.key==='ArrowRight'){e.preventDefault();show(at+1);}});
 return wrap;
}
function numbersTable(keys,L,key){
 // Each value carries its actual measurement step; missing later data is never back-dated.
 const det=el('details','numbers');det.dataset.key=key;det.append(el('summary','','Show the numbers per segment'));
 const t=el('table','results-table'),head=el('tr');head.append(el('th','','Segment'));
 for(const k of keys)head.append(el('th','',shortName(L.defs[k])));t.append(head);
 for(const s of L.segments||[]){let idx=-1;L.x.segment.forEach((g,i)=>{if(g===s.index)idx=i;});if(idx<0)continue;
  const row=el('tr');row.append(el('td','',s.label));
  for(const k of keys){let v=null,measuredAt=null,firstStep=null;for(let i=idx;i>=0&&L.x.segment[i]===s.index;i--){const a=L.series[k];if(a&&a[i]!=null){v=a[i];measuredAt=L.x.step[i];firstStep=L.x.start_step?.[i];break;}}row.append(el('td','',dtext(L.defs[k],disp(L.defs[k],v))+(measuredAt===null?'':firstStep!=null?' (average over steps '+stepText(firstStep)+' to '+stepText(measuredAt)+')':' at step '+stepText(measuredAt))));}
  t.append(row);}
 const sc=el('div','scroll');sc.append(t);det.append(sc);return det;
}
function probePanels(item,L){
 const P=L.probes||{},entries=(P.items||{})[String(item)]||[];
 if(!entries.length)return [el('p','unmeasured-note',P.index_found?'Unmeasured: the probe index holds no result for this item on this lineage’s checkpoints.':'Unmeasured: no probe has run on this lineage’s checkpoints yet; the probe suite writes Saved/mimo_bridge/gof_probes/index.jsonl.')];
 return entries.slice(0,4).map(e=>{
  const box=el('div','gauge-panel'),h=e.latest||{},w=e.words||{};box.append(el('h4','',(w.label||e.probe||'probe')+(w.stand_in?' (a stand-in, not this item’s measurement)':'')));
  if(Number.isFinite(h.lineage_step)){
   const before=Number.isFinite(state?.training_brief?.inherited_updates)&&h.lineage_step<=state.training_brief.inherited_updates;
   box.append(el('p','test-scope','Saved-model test at update '+Math.trunc(h.lineage_step).toLocaleString()+(h.time?' (recorded '+h.time+')':'')+'. '+(before?'This predates the current restart; the current version has not been tested here.':'This is a saved-model result, not a test of the changing live model.')));
  }
  box.append(el('p','gauge-latest',w.sentence||'The probe’s headline could not be read.'));
  // w.failed: the probe's newest run on this lineage produced no reading (the server's sentence)
  for(const t of [w.scale&&('Scale: '+w.scale),w.goal,w.failed])if(t)box.append(el('p','small muted',t));
  const pts=(e.points||[]).filter(p=>p[0]!=null&&p[1]!=null);
  if(pts.length&&w.share!=null){const pseudo={x:{step:pts.map(p=>p[0]),time:pts.map(()=>null),segment:pts.map(()=>-1)},segments:L.segments,
   series:{p:pts.map(p=>p[1])},defs:{p:{title:w.label||e.probe||'probe',lo:[0,''],hi:[1,''],display:{kind:'per100',a:100,b:0,unit:'%'},floor:[0,'floor']}}};
   box.append(chart(['p'],pseudo));}
  if(w.own){const d=el('details','gof-own');d.append(el('summary','','The probe’s own words'),el('p','small',w.own));box.append(d);}
  return box;});
}
async function loadMore(){
 const box=$('more-gauges-body');if(!box)return;
 const now=Date.now()/1000;if(moreData&&now-moreAt<300){renderMore();return;}
 box.replaceChildren(el('p','small muted','Reading every gauge of the lineage…'));
 try{const r=await fetch('/api/lineage?all=1',{cache:'no-store',signal:AbortSignal.timeout(90000)});if(!r.ok)throw Error('the server answered '+r.status);moreData=await r.json();moreAt=now;}
 catch(e){box.replaceChildren(el('p','unmeasured-note','The other gauges could not be read ('+e.message+').'));return;}
 renderMore();
}
function renderMore(){
 const box=$('more-gauges-body'),L=moreData;if(!box||!L)return;
 const panels=L.more||[];
 if(!panels.length){box.replaceChildren(el('p','small muted',RESTART_NOTE));return;}
 box.replaceChildren(...panels.map(p=>{const a=el('article','gauge-card');a.append(gaugePanel(p,L,'more'));return a;}));
}
const moreBox=$('more-gauges');if(moreBox)moreBox.addEventListener('toggle',()=>{if(moreBox.open)loadMore();});

// ---------------------------------------------------------------- the Development view
// GET /api/observer (tools/observer/view.py): the developmental observer's six stages per capability,
// the finger rows, the body map, the timeline, novelty candidates and the regression watch; GET
// /api/observer/history: the dated copies, their rule changes and each row's counts for the small charts.
// Every sentence is built by the server; the page lays them out.
let observer=null,observerAt=0,observerError='',obsHistory=null,obsHistoryAt=0,obsShow='now',obsCompare='',obsRequest=0;
// ?at= shows a dated copy of the report and ?compare= another one beside it (tools/observer/history.py)
async function loadObserver(force=false){
 const now=Date.now()/1000;if(!force&&observer&&now-observerAt<60)return;observerAt=now;
 // only the newest request is shown: a slow answer for an earlier choice in the date list is dropped
 const mine=++obsRequest;
 const q=new URLSearchParams();if(obsShow&&obsShow!=='now')q.set('at',obsShow);if(obsCompare)q.set('compare',obsCompare);
 try{const r=await fetch('/api/observer'+(q.toString()?'?'+q:''),{cache:'no-store',signal:AbortSignal.timeout(30000)});
  if(mine!==obsRequest)return;
  if(r.status===404){observerError=RESTART_NOTE;renderObserver();return;}
  if(!r.ok)throw Error('the server answered '+r.status);observer=await r.json();observerError='';}
 catch(e){if(mine!==obsRequest)return;observerError='The observer report could not be read ('+e.message+'); the page retries every minute while this tab is open.';}
 renderObserver();
 // the dated copies live on a USB disk that can take 10 s to wake: the report is shown first, the charts after
 if(await loadObserverHistory()&&mine===obsRequest)renderObserver();
}
async function loadObserverHistory(force=false){
 const now=Date.now()/1000;if(!force&&obsHistory&&now-obsHistoryAt<300)return false;obsHistoryAt=now;
 try{const r=await fetch('/api/observer/history',{cache:'no-store',signal:AbortSignal.timeout(30000)});if(r.ok){obsHistory=await r.json();return true;}}
 catch(e){/* the dated copies are optional: the page still shows now */}
 return false;
}
const OBS_STAGE_KEYS=['observed_once','repeated','predicted_before_action','selected_intentionally','generalized','retained'];
function obsStrip(stages,label){
 const strip=el('div','obs-strip');strip.setAttribute('aria-label','Stages for '+label);
 // each cell: the stage, its marker (passed / not yet / not judged yet / not measurable) and the detail;
 // st.tip (view.stage_tip): the stage's chance against the line it was judged by, as the cell's tooltip
 for(const st of stages||[]){const c=el('div','obs-cell '+st.state);const b=el('b','',st.title);if(st.question)b.title=st.question;
  c.append(b,el('span','obs-marker',st.marker||''),el('span','obs-detail',st.detail||st.text));c.title=st.tip||st.text||'';strip.append(c);}
 return strip;
}
function obsChart(key){
 // the row's occasions against random flailing's count for the same time at every saved date and now
 const h=obsHistory,pts=h?.series?.[key];if(!h||!pts||pts.length<2)return null;
 // saved dates evenly spaced (training pauses would otherwise squeeze them together); each point's tooltip names its date
 const E=h.entries||[],W=280,H=60,L=38,R=6,T=6,B=4;
 const ts=pts.map(p=>E[p[0]].time),t0=Math.min(...ts),t1=Math.max(...ts),ix=pts.map(p=>p[0]),i0=Math.min(...ix),i1=Math.max(...ix);
 const top=Math.max(1,...pts.map(p=>Math.max(p[1]||0,p[2]||0)));
 const xi=i=>L+(W-L-R)*(i-i0)/Math.max(1,i1-i0),Y=v=>T+(H-T-B)*(1-v/top);
 // a rule change sits between the last saved date before it and the first after it
 const xRule=t=>{const after=E.findIndex(e=>e.time>=t);return after<=i0?null:xi(after-0.5);};
 const svg=sv('svg',{viewBox:'0 0 '+W+' '+H,class:'obs-chart',role:'img'});
 const breaks=new Set(h.breaks||[]);
 // a new segment after a missing value or across a change of counting rules
 const path=j=>{let d='',gap=true;for(const p of pts){if(p[j]==null){gap=true;continue;}d+=(gap||breaks.has(p[0])?'M':'L')+xi(p[0]).toFixed(1)+','+Y(p[j]).toFixed(1);gap=false;}return d;};
 for(const r of h.rule_changes||[])if(r.time>t0&&r.time<=t1){const x=xRule(r.time);if(x==null)continue;const line=sv('line',{x1:x,x2:x,y1:T-4,y2:H-B,class:'obs-chart-rule'});line.append(sv('title',{},r.label+': the points before and after it are not like for like.'));svg.append(line);}
 svg.append(sv('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'obs-chart-axis'}));
 svg.append(sv('path',{d:path(2),class:'obs-chart-flail'}),sv('path',{d:path(1),class:'obs-chart-own'}));
 for(const p of pts){if(p[1]==null)continue;const e=E[p[0]];const c=sv('circle',{cx:xi(p[0]),cy:Y(p[1]),r:e.rebuilt?3:3.5,class:e.rebuilt?'obs-chart-dot rebuilt':'obs-chart-dot'});
  c.append(sv('title',{},e.label+': '+Number(p[1]).toLocaleString('en-US')+' occasions; random flailing would make '+(p[2]==null?'an unmeasured number':Number(p[2]).toLocaleString('en-US',{maximumFractionDigits:p[2]<100?1:0}))+' in the same time.'));svg.append(c);}
 svg.append(sv('text',{x:L-4,y:H-B,class:'obs-chart-label end'},'0'),sv('text',{x:L-4,y:T+6,class:'obs-chart-label end'},Number(top).toLocaleString('en-US',{maximumFractionDigits:0})));
 const fig=el('figure','obs-figure');const cap=el('figcaption','small muted');
 cap.append(document.createTextNode(clockAt(t0)+' to '+(E[i1].file==='now'?'now':clockAt(t1))+': '));
 cap.append(el('span','obs-key own','—'),document.createTextNode(' occasions  '),el('span','obs-key flail','- -'),document.createTextNode(' random flailing in the same time'+((h.rule_changes||[]).some(r=>r.time>t0&&r.time<=t1)?'  ':'')));
 if((h.rule_changes||[]).some(r=>r.time>t0&&r.time<=t1))cap.append(el('span','obs-key rule','┊'),document.createTextNode(' counting rules changed'));
 fig.append(svg,cap);return fig;
}
function obsRow(row){
 // the catalog code (H05, HX1...) stays out of the reading line: it is the row head's tooltip
 const d=el('div','obs-row');const head=el('div','obs-row-head');head.append(el('strong','',row.name||row.label));head.title=row.id||row.cap||'';d.append(head);
 if(row.lead)d.append(el('p','obs-lead',row.lead));
 for(const c of row.changes||[])d.append(el('p','obs-change',c));
 const chart=obsChart(row.key);if(chart)d.append(chart);
 if(row.summary)d.append(el('p','small',row.summary));
 // row.wellness (view.wellness_view): one plain line, its numbers in the tooltip
 if(row.wellness?.text){const w=el('p','small',row.wellness.text);if(row.wellness.tip)w.title=row.wellness.tip;d.append(w);}
 for(const n of row.notes||[])d.append(el('p','obs-note neutral',n));
 if(row.regressed_text)d.append(el('p','obs-note',row.regressed_text));
 d.append(obsStrip(row.stages,row.name||row.label));
 if(row.no_overseer){const x=el('details');x.append(el('summary','small',observer?.no_overseer_label||'With the caregiver away'),obsStrip(row.no_overseer,(row.name||'')+' with the caregiver away'));d.append(x);}
 if(row.subs?.length){const x=el('details');x.append(el('summary','small','By '+(row.subs.length===1?'part':row.subs.length+' parts')));
  for(const s of row.subs){const p=el('p','small');p.append(el('strong','',s.label),document.createTextNode(' · '+Number(s.counted||0).toLocaleString('en-US')+' detected. '+(s.lead||'')));x.append(p);for(const c of s.changes||[])x.append(el('p','obs-change',c));x.append(obsStrip(s.stages,s.label));}
  d.append(x);}
 if(row.replay){const a=el('a','small','Replay the first detected act (up to 10 s; other acts may appear) ↗');a.href=row.replay;d.append(a);}
 return d;
}
function obsVisitRows(o){
 const m={};for(const g of o.groups||[])for(const r of g.rows||[]){if(r.track)m[r.key]=[r.name,r.track];for(const s of r.subs||[])if(s.track)m[s.key]=[r.name+' ('+s.label+')',s.track];}
 return m;
}
function obsStore(kind,key,value){
 // per-viewer memory of the last visit; any storage failure leaves the box saying so
 try{const s=kind==='session'?sessionStorage:localStorage;if(value===undefined)return s.getItem(key);s.setItem(key,value);return value;}catch(e){return null;}
}
function renderVisit(o){
 const box=$('obs-visit');if(!box)return;box.replaceChildren();
 if(o.shown?.kind!=='live'){box.hidden=true;return;}box.hidden=false;
 const current=JSON.stringify({time:o.generated,step:o.last_step,rows:obsVisitRows(o)});
 // a new browser session moves "last visit" to "previous visit"; refreshes within a session keep comparing with it
 if(!obsStore('session','obs.visit.session')){const last=obsStore('local','obs.visit.last');if(last)obsStore('local','obs.visit.previous',last);obsStore('session','obs.visit.session','1');}
 const prevText=obsStore('local','obs.visit.previous');obsStore('local','obs.visit.last',current);
 box.append(el('h2','','What changed since your last visit'));
 let prev=null;try{prev=prevText?JSON.parse(prevText):null;}catch(e){prev=null;}
 if(!prev){box.append(el('p','small muted','This is the first visit this browser remembers. Next time, this box lists the stages each behavior gained or lost since now.'));return;}
 const titles=Object.fromEntries((o.stages||[]).map(s=>[s.key,s.title]));const now=obsVisitRows(o),lines=[];
 for(const [key,[name,t]] of Object.entries(now)){const before=prev.rows?.[key];if(!before)continue;const was=before[1][2]||0,is=t[2]||0;
  const gained=OBS_STAGE_KEYS.filter((k,i)=>(is>>i&1)&&!(was>>i&1)).map(k=>titles[k]),lost=OBS_STAGE_KEYS.filter((k,i)=>(was>>i&1)&&!(is>>i&1)).map(k=>titles[k]);
  if(gained.length)lines.push([name+': passed '+gained.join(' and ')+'.','gain']);if(lost.length)lines.push([name+': no longer passes '+lost.join(' and ')+'.','loss']);}
 box.append(el('p','small muted','Your last visit saw the report of '+(prev.time?clockAt(prev.time):'an unrecorded time')+' (step '+stepText(prev.step)+').'));
 if(!lines.length)box.append(el('p','','No behavior gained or lost a stage since then.'));
 for(const [t,cls] of lines)box.append(el('p','obs-change '+cls,t));
}
function renderObsControls(o){
 const box=$('obs-dates');if(!box)return;const h=obsHistory;box.replaceChildren();
 const entries=(h?.entries||[]).slice().reverse();
 const show=el('select');show.id='obs-show';const cmp=el('select');cmp.id='obs-compare';
 for(const e of entries){const v=e.file==='now'?'now':e.file;show.append(new Option(e.file==='now'?'Now: '+e.label.replace(/^now \(|\)$/g,''):e.label,v));}
 if(!entries.length)show.append(new Option('Now','now'));
 // a copy deleted after it was chosen stays named in the list until another is chosen
 for(const v of [obsShow,obsCompare])if(v&&v!=='now'&&!entries.some(e=>e.file===v)){show.append(new Option(v+' (no longer listed)',v));}
 show.value=obsShow;cmp.append(new Option('Nothing',''));for(const e of entries){const v=e.file==='now'?'now':e.file;if(v!==obsShow)cmp.append(new Option(e.file==='now'?'Now':e.label,v));}cmp.value=obsCompare;
 show.addEventListener('change',()=>{obsShow=show.value;if(obsCompare===obsShow)obsCompare='';loadObserver(true);});
 cmp.addEventListener('change',()=>{obsCompare=cmp.value;loadObserver(true);});
 const l1=el('label','','Show the report of '),l2=el('label','','Compare with ');l1.append(show);l2.append(cmp);box.append(l1,l2);
 const notes=el('div','obs-date-notes');
 if(h?.note)notes.append(el('p','small muted',h.note));
 else if(h?.entries)notes.append(el('p','small muted','A dated copy is saved every 4 hours and kept 5 days. Copies marked "rebuilt later" were computed afterwards from the observer\'s records, under today\'s counting rules.'));
 if(o.shown?.note)notes.append(el('p','small',o.shown.note));
 if(o.comparison?.text)notes.append(el('p','obs-compare-head',o.comparison.text));
 if(o.comparison?.warning)notes.append(el('p','obs-note',o.comparison.warning));
 box.append(notes);
}
function renderObserver(){
 const o=observer;
 if(!o){$('obs-headline').textContent=observerError||'Reading the observer’s report.';return;}
 renderObsControls(o);
 if(!o.available){$('obs-headline').textContent=o.sentence||observerError;
  // nothing of an earlier report stays on screen under a report that could not be read
  for(const id of ['obs-visit','obs-stage-key','obs-markers','obs-map','obs-fingers','obs-groups','obs-timeline','obs-novelty','obs-regressions'])$(id)?.replaceChildren();
  return;}
 renderVisit(o);
 $('obs-headline').textContent=(o.shown?.kind==='copy'?'Showing the dated copy of '+o.shown.label+'.\n':'')+(o.headline_lines||[o.headline]).join('\n');
 const sv=o.service||{};const written=o.generated?' · report written '+new Date(o.generated*1000).toLocaleTimeString([],{hour:'numeric',minute:'2-digit'}):'';
 $('obs-service').textContent=(sv.running?'The observer’s digester last reported '+age(sv.last_cycle_age_s)+' ago':'The observer’s digester is not reporting')+written+(sv.error?' · its last cycle failed: '+sv.error:'')+(observerError?' · '+observerError:'');
 const key=$('obs-stage-key');key.replaceChildren();
 for(const st of o.stages||[]){const b=el('button','');b.type='button';b.append(el('strong','',st.title),el('span','',st.question||''));b.addEventListener('click',()=>goalDialog(st.title+': '+(st.question||''),[st.meaning,'A stage without its control (for example before random flailing is recorded for this world) reads not measurable, never passed or failed.']));key.append(b);}
 const mk=$('obs-markers');if(mk){mk.replaceChildren(document.createTextNode((o.full_behaviour||'')+' Each cell says: '));for(const [state,word] of Object.entries(o.markers||{})){mk.append(el('span','obs-marker-key '+state,word),document.createTextNode(' '));}}
 const map=$('obs-map');map.replaceChildren();for(const r of o.body_map||[]){const a=el('article',r.stage?'seen':'');a.append(el('h3','',r.region),el('p','',r.text));map.append(a);}
 const fingers=$('obs-fingers');fingers.replaceChildren(...(o.fingers||[]).map(obsRow));
 const groups=$('obs-groups');groups.replaceChildren();for(const g of o.groups||[]){const d=el('details');d.append(el('summary','',g.summary||(g.title+' · '+g.rows.length+' capabilities')));for(const r of g.rows)d.append(obsRow(r));groups.append(d);}
 const tl=$('obs-timeline');tl.replaceChildren();if(!(o.timeline||[]).length)tl.append(el('li','small muted','No stage has been reached yet.'));for(const t of o.timeline||[])tl.append(el('li',t.lost?'obs-lost':(t.note?'small muted':''),t.text));
 const nov=$('obs-novelty');nov.replaceChildren();if(!(o.novelty||[]).length)nov.append(el('p','small muted','No recurring unexplained pattern yet (a candidate needs 5 separate episodes on 2 or more bodies; an episode is one body part\'s stretch of activity, ending after half a second of quiet, or two parts\' episodes starting within half a second of each other).'));for(const c of o.novelty||[]){const p=el('p','obs-novel',c.text);if(c.label)p.title=c.label;if(c.replay){const a=el('a','small',' Replay ↗');a.href=c.replay;p.append(a);}nov.append(p);}
 const reg=$('obs-regressions');reg.replaceChildren();if(!(o.regressions||[]).length)reg.append(el('p','small muted',o.regression_note||'Not checked yet.'));for(const r of o.regressions||[])reg.append(el('p','obs-note',r.text));
 for(const [id,list] of [['obs-horizon',o.horizon],['obs-fast',o.fast],['obs-health',o.health]]){const box=$(id);box.replaceChildren(...(list||[]).map(t=>el('p','small',t)));if(!(list||[]).length)box.append(el('p','small muted','Nothing recorded yet.'));}
 $('obs-baseline').textContent=o.baseline||'';
 const roles=$('obs-roles');roles.replaceChildren();if(o.roles?.rule)roles.append(el('p','small',o.roles.rule));
 for(const r of o.roles?.roles||[]){const p=el('p','small');p.append(el('strong','',r.role+'. '),document.createTextNode((r.note||'')+' Treatment: '+r.treatment+'.'+((r.fields||[]).length?' Fields: '+r.fields.map(f=>typeof f==='string'?f:(f.field||'')+(f.why?' ('+f.why+')':'')).join('; ')+'.':'')));roles.append(p);}
 const links=$('obs-links');links.replaceChildren();for(const [label,key] of [['Design and decisions','design'],['Capability catalog','catalog'],['Validation report','validation'],['Full report','report']]){if(o.links?.[key]){const a=el('a','',label+' ↗');a.href=o.links[key];links.append(a);}}
 // ?expand opens every section of this view at once (reading everything, or a full-page check);
 // ?expand=sections opens only the view's own sections, not each capability's parts
 const expand=new URLSearchParams(location.search).get('expand');
 if(expand!==null)for(const d of $('observer').querySelectorAll(expand==='sections'?':scope > details':'details'))d.open=true;
}
setInterval(()=>{if(!$('observer').hidden&&obsShow==='now')loadObserver();},60000);

// ---------------------------------------------------------------- the Tests page
// ---------------------------------------------------------------- the Tests page
// GET /api/tests?view=sections (tools/test_registry.py dashboard_view, schema 2): every row of
// docs/gain_of_function_v4/test_registry.json, read from the file on each request, placed in one of seven sections
// the server computes from the runner's record (QUEUED, RUNNING, BLOCKED, FAILED, INVALID, STOPPED, DONE), with the
// audit of live test processes against the running rows (both directions) and "The names used on this page". Every
// sentence is built by the server; the page lays them out (the recovery directive of Oct 5, 2026, sections 7A and 8):
//  - "The names used on this page" first: the seven sections with their meanings, the outcomes, resource classes and
//    process states the rows show, the components the rows name, and the labels of the cards themselves;
//  - the errors and the audit's sentence above the sections;
//  - seven collapsible sections: on the first load only the ones the server marks open_by_default (QUEUED) are open;
//    afterwards the page keeps what the reader opened or closed across the 30-second reloads, never across loads;
//  - one card per row: question, component (a link that opens that component in the Architecture tab of the
//    explainer), state, age, direct target, conclusion and next action, and for a row whose section is not DONE its
//    diagram (input, operation, measurement, decision); the rest of the row, the runner's record and the raw
//    result sit behind Details, which is drawn when it is opened.
// ---------------------------------------------------------------- the aliases in data text (finding F7)
// A row's own text, the supervisor's pause reason and the other records are printed as they were written. Each
// forbidden alias of the component manifest in them (GET /api/names, forbidden_aliases) is followed by its mark
// "[alias of ID ...]" naming the ids it has named, so the reader sees the canonical name beside it; the page's own
// fixed text is never marked (the reply check refuses an alias there). The live check reads the marked text with the
// reply check's marked_ok (the dashboard review of Oct 5, 2026, finding F7).
let aliasList=null,aliasPattern=null;
const ALIAS_CONTAINERS=['tests-groups','tests-errors-list','tests-warnings','tests-gaps','tests-names-body','tests-investigations-body',
 'mode-sentence','mode-more','scoreboard-stages','scoreboard-shared','top-blockers','blockers-sentence','status-lines-list',
 'overview-errors-list','week-pauses-list'];
const escapeRe=s=>String(s).replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
async function loadAliases(){
 try{const r=await fetch('/api/names',{cache:'no-store',signal:AbortSignal.timeout(30000)});if(!r.ok)return;const d=await r.json();
  aliasList=(d.forbidden_aliases||[]).filter(a=>a&&a.alias).map(a=>({alias:String(a.alias).toLowerCase(),ids:[...new Set(a.canonical||[])].sort()}))
   .sort((a,b)=>b.alias.length-a.alias.length);
  aliasPattern=aliasList.length?new RegExp('(?<![A-Za-z0-9_\\-])('+aliasList.map(a=>a.alias.split(/\s+/).map(escapeRe).join('[\\s-]+')).join('|')+')s?(?![A-Za-z0-9_])','gi'):null;
  markAllAliases();}
 catch(e){}
}
function aliasOwners(found){
 const low=found.toLowerCase().replace(/[\s-]+/g,' ').replace(/s$/,'');
 const hit=(aliasList||[]).find(a=>a.alias===low||a.alias===found.toLowerCase().replace(/[\s-]+/g,' '));
 return hit?hit.ids:[];
}
function markAliases(root){
 if(!root||!aliasPattern)return;
 const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,{acceptNode:n=>{
  const p=n.parentElement;if(!p||p.closest('code,script,style,.alias-mark'))return NodeFilter.FILTER_REJECT;return NodeFilter.FILTER_ACCEPT;}});
 const nodes=[];while(walker.nextNode())nodes.push(walker.currentNode);
 for(const node of nodes){
  const text=node.nodeValue;aliasPattern.lastIndex=0;const hits=[...text.matchAll(aliasPattern)];if(!hits.length)continue;
  const frag=document.createDocumentFragment();let at=0;
  for(const m of hits){
   const end=m.index+m[0].length,ids=aliasOwners(m[0]);if(!ids.length)continue;
   if(end===text.length&&node.nextSibling&&node.nextSibling.classList&&node.nextSibling.classList.contains('alias-mark'))continue;
   frag.append(document.createTextNode(text.slice(at,end)));
   const mark=el('span','alias-mark',' [alias of '+(ids.length>1?ids.slice(0,-1).join(', ')+' or '+ids[ids.length-1]:ids[0])+']');
   mark.title='An alias the record itself wrote; the canonical name is given in the brackets';frag.append(mark);at=end;
  }
  if(!at)continue;frag.append(document.createTextNode(text.slice(at)));node.replaceWith(frag);
 }
}
function markAllAliases(){for(const id of ALIAS_CONTAINERS)markAliases($(id));}
loadAliases();setInterval(loadAliases,300000);

let testsData=null,testsError='',testsRequest=0;
const testsSectionOpen=new Map();            // section key -> open, as the reader left it during this page's life
const testsLongOpen=new Set();               // the long texts a reader asked to see whole
const SECTION_TONE={QUEUED:'queued',RUNNING:'running',BLOCKED:'blocked',FAILED:'failed',INVALID:'failed',STOPPED:'stopped',DONE:'done'};
const DIAGRAM_PARTS=[['input','Input'],['operation','Operation'],['measurement','Measurement'],['decision','Decision']];
// the labels of the cards, defined in "The names used on this page"
const CARD_TERMS=[
 ['Age','since the run started (a row in RUNNING), since its last run ended (after a run), else since the row was written'],
 ['Direct target','the real quantity the test should improve, with its two ends (the row’s target_metric field)'],
 ['Conclusion','the outcome of the last run in the runner’s record, in words; a verdict typed into the row is shown here only when it holds no number, because a number without its two judgements stays under Details'],
 ['Next action','what happens next: the follow-up written for the last run, else the next action the runner wrote at that run’s end, else one sentence for the row’s section'],
 ['Diagram','four phrases of the row, shown on every card whose section is not DONE: what goes in (input), what is computed (operation), what is measured (measurement), and what happens on pass and on fail (decision)']];
const JUDGEMENT_TERMS=[
 ['Statistically clear?','whether a reading’s difference from its baseline is told apart from run-to-run noise: yes, no, or not judged, with the basis of a yes or a no'],
 ['Large enough to matter for gain of function?','whether the difference is as large as the row’s material line for gain of function: yes, no, or GOF threshold not yet defined']];
const testKey=v=>(v.id||'')+'|'+v.position;
function explainerHref(component){
 const base=String(state?.explainer?.url||'/files/Saved/mimo_bridge/explainer.html').split('#')[0];
 return base+'#t=arch&component='+encodeURIComponent(component);
}
// ---------------------------------------------------------------- root-cause investigations (the Tests page section)
// GET /api/investigations (tools/elimination.py view): every investigation file of
// docs/gain_of_function_v4/investigations, each a live process-of-elimination diagram whose boxes the server computes
// from the test registry's rows and their result files on each request (the style guide, the operator's rule of Oct 5, 2026).
// Read with the Tests page and every 30 seconds while the tab is open; drawn by elimination_diagram.js, the renderer the
// explainer's Architecture tab uses for the same data. The details a reader opened stay open across the reloads.
let invData=null,invError='',invRequest=0;
const invOpen=new Set();
async function loadInvestigations(){
 const mine=++invRequest;
 try{const r=await fetch('/api/investigations',{cache:'no-store',signal:AbortSignal.timeout(30000)});
  if(mine!==invRequest)return;
  if(r.status===404||r.status===503){invError=RESTART_NOTE;renderInvestigations();return;}
  if(!r.ok)throw Error('the server answered '+r.status);invData=await r.json();invError='';}
 catch(e){if(mine!==invRequest)return;invError='The investigations could not be read ('+e.message+'); the page tries again every 30 seconds while this tab is open.';}
 renderInvestigations();
}
function investigationHref(id){
 const base=String(state?.explainer?.url||'/files/Saved/mimo_bridge/explainer.html').split('#')[0];
 return base+'#t=arch&investigation='+encodeURIComponent(id);
}
function renderInvestigations(){
 const box=$('tests-investigations-body'),st=$('tests-investigations-status');if(!box||!st)return;
 if(!invData){st.textContent=invError||'Reading the investigations.';st.hidden=false;return;}
 const list=invData.investigations||[];
 st.textContent=invError||(list.length?'':'No investigation file is in '+(invData.folder||'docs/gain_of_function_v4/investigations')+'.');
 st.classList.toggle('error',Boolean(invError));st.hidden=!st.textContent;
 if(!window.EliminationDiagram){st.textContent='The diagram renderer (elimination_diagram.js) did not load.';st.hidden=false;st.classList.add('error');return;}
 const y=window.scrollY;box.replaceChildren();
 if((invData.errors||[]).length){const ul=el('ul','tests-error-list');for(const e of invData.errors){const li=el('li');li.append(el('strong','','investigation: '),document.createTextNode(e.text||''));ul.append(li);}box.append(ul);}
 EliminationDiagram.renderAll(box,invData,{href:explainerHref,onRow:row=>showTestCard(row),
  sourceText:'Read from the development dashboard at '+(invData.read_at_text||'a time not recorded')+'.'});
 for(const sec of box.querySelectorAll('section.elim')){
  const id=sec.dataset.investigation,p=el('p','small');const a=el('a','','Open this investigation in the Architecture tab of How the model works');
  a.href=investigationHref(id);p.append(a);sec.querySelector('.elim-title')?.after(p);
 }
 box.querySelectorAll('details').forEach((d,i)=>{const host=d.closest('[id]');const key=(host?host.id:'')+'|'+d.className+'|'+i;
  d.dataset.key=key;if(invOpen.has(key))d.open=true;d.addEventListener('toggle',()=>{if(d.open)invOpen.add(key);else invOpen.delete(key);});});
 if(testsData)namesBlock(testsData);
 markAliases(box);
 if(Math.abs(window.scrollY-y)>1)window.scrollTo(0,y);
}
async function loadTests(){
 loadInvestigations();
 const mine=++testsRequest;
 try{const r=await fetch('/api/tests?view=sections',{cache:'no-store',signal:AbortSignal.timeout(30000)});
  if(mine!==testsRequest)return;
  if(r.status===404){testsError=RESTART_NOTE;renderTests();return;}
  if(!r.ok)throw Error('the server answered '+r.status);testsData=await r.json();testsError='';}
 catch(e){if(mine!==testsRequest)return;testsError='The test registry could not be read ('+e.message+'); the page tries again every 30 seconds while this tab is open.';}
 renderTests();
}
// headings and titles carry data-plain and file paths sit in <code>: the glossary (development_dashboard_terms.js)
// adds its term buttons to the rows' prose only
function testSection(s){
 const box=el('section','test-section'),h=el('h4','',s.label);h.dataset.plain='';box.append(h);
 const list=el(s.ordered?'ol':'ul',s.code?'test-code':'');
 for(const line of s.lines||[]){const li=el('li');li.append(s.code?el('code','',line):document.createTextNode(line));list.append(li);}
 box.append(list);return box;
}
// a text the card shows in at most three lines until the reader asks for the whole of it; nothing is cut out of the
// text itself: the whole text is in the element and the button only lifts the three-line limit
function longText(tag,cls,text,key,limit=260){
 const node=el(tag,cls,text);
 if(String(text||'').length<=limit)return [node];
 const whole=testsLongOpen.has(key);node.classList.toggle('clamped',!whole);
 const b=el('button','text-button long-toggle',whole?'Show less':'Show all of it');b.type='button';b.setAttribute('aria-expanded',String(whole));
 b.addEventListener('click',()=>{const open=node.classList.toggle('clamped');b.textContent=open?'Show all of it':'Show less';b.setAttribute('aria-expanded',String(!open));if(open)testsLongOpen.delete(key);else testsLongOpen.add(key);});
 return [node,b];
}
function componentLink(v){
 const a=v.architecture,id=v.card?.component||v.component;
 if(!id){const s=el('span','test-component missing','No component named');s.title='The row names no component of the component manifest';return s;}
 if(a&&a.known===false){const s=el('span','test-component unknown',id);s.title='Not an id of the component manifest (listed under Errors)';return s;}
 const link=el('a','test-component');link.href=explainerHref(id);link.dataset.component=id;
 link.title='Opens '+id+' in the Architecture tab of How the model works';
 link.append(el('span','test-component-id',id));if(a?.name)link.append(el('span','test-component-name',a.name));
 return link;
}
function diagramStrip(dg){
 const box=el('div','test-diagram');box.setAttribute('role','group');box.setAttribute('aria-label','Diagram of the test: input, operation, measurement, decision');
 DIAGRAM_PARTS.forEach(([key,label],i)=>{
  if(i){const arrow=el('span','diagram-arrow','→');arrow.setAttribute('aria-hidden','true');box.append(arrow);}
  const part=el('div','diagram-box');part.dataset.part=key;part.append(el('span','diagram-label',label),el('span','diagram-text',dg[key]));box.append(part);
 });
 return box;
}
function readingTable(readings){
 const t=el('table','reading-table'),head=el('tr');
 for(const h of ['Reading','Value','Baseline','Statistically clear?','Large enough to matter for gain of function?'])head.append(el('th','',h));
 const body=el('tbody');body.append(head);
 for(const r of readings){
  const row=el('tr'),name=el('td');name.append(el('strong','',r.name||'a reading'));if(r.meaning)name.append(el('span','reading-meaning',r.meaning));
  const clear=el('td','judgement '+(r.statistically_clear==='yes'?'yes':r.statistically_clear==='no'?'no':'open'),r.statistically_clear||'not judged');
  if(r.statistical_basis)clear.append(el('span','reading-meaning',r.statistical_basis));
  const gof=el('td','judgement '+(r.gof_material==='yes'?'yes':r.gof_material==='no'?'no':'open'),r.gof_material||'GOF threshold not yet defined');
  row.append(name,el('td','',[r.value,r.unit].filter(x=>x!==null&&x!==undefined&&x!=='').join(' ')||'no value'),el('td','',r.baseline||''),clear,gof);body.append(row);
  for(const p of r.problems||[]){const tr=el('tr','reading-problem'),td=el('td','','This reading is not in its declared form: '+p+'.');td.colSpan=5;tr.append(td);body.append(tr);}
 }
 t.append(body);return t;
}
function runRecord(run){
 const dl=el('dl','run-record');
 const add=(k,v)=>{if(v===null||v===undefined||v==='')return;dl.append(el('dt','',k),el('dd','',String(v)));};
 add('Outcome in the runner’s record',run.state);add('Started',run.started);add('Ended',run.ended);add('Exit code',run.exit_code);
 add('Commit',run.commit?(run.commit+(run.commit_dirty?' (with changes not yet committed)':'')):null);add('Checkpoint',run.checkpoint);
 add('Owner',run.owner);add('Resource class',run.resource_class);add('Device',run.device);add('Runner',run.runner);add('Judged by',run.judged_by);
 add('Result file',run.result_file);add('Log',run.log);
 add('Written by this module’s runner',run.sealed?'yes: its seal verifies':'no: a runner of the older kind or a hand edit wrote this record');
 return dl;
}
function detailsBody(v){
 const body=el('div','test-details-body');
 const line=(label,text,cls)=>{if(!text)return;const p=el('p',cls||'');p.append(el('strong','',label+': '),document.createTextNode(text));body.append(p);};
 line('Title',v.title);
 if(v.id){const p=el('p');p.append(el('strong','','Job name: '),el('code','',v.id));body.append(p);}
 line('Why it matters',v.why);
 for(const t of [v.incomplete_text,v.refused_text])if(t)body.append(el('p','ability-results-warning',t));
 line('Why it is in this section',v.section_reason);
 if(v.process&&v.process.state&&v.process.state!=='RUNNING')line('Process state',v.process.state+(v.process.held_text?': '+v.process.held_text:''));
 if(v.gaps?.length)line('Fields of the directive’s section 6 this row does not fill',v.gaps.join(', '));
 if(v.readings?.length){body.append(el('h4','details-head','Readings of the last run, each with its two judgements'));body.append(readingTable(v.readings));}
 if(v.unjudged_result)body.append(el('p','raw-note','Raw result below: its numbers carry no judgement of statistical clarity and none of size for gain of function.'));
 if(v.result_from_earlier_run)body.append(el('p','raw-note','The result lines below come from an earlier run of this row; its last run wrote no result.'));
 for(const t of [v.last_run_text,v.result_file_text])if(t)body.append(el('p','small muted',t));
 if(v.run){body.append(el('h4','details-head','The runner’s record of the last run'));body.append(runRecord(v.run));}
 for(const s of v.sections||[])body.append(testSection(s));
 return body;
}
function testDetails(v,open){
 const d=el('details','test-details');d.dataset.key=testKey(v);
 d.append(el('summary','','Details'));
 const fill=()=>{if(d.open&&!d.dataset.filled){d.dataset.filled='1';const b=detailsBody(v);d.append(b);markAliases(b);}};
 d.addEventListener('toggle',fill);d.open=open.has(d.dataset.key);fill();
 return d;
}
function testCard(v,open){
 const c=v.card||{},section=v.section||'BLOCKED';
 const a=el('article','test-card tone-'+(SECTION_TONE[section]||'blocked'));a.dataset.row=v.id||'';a.dataset.section=section;
 a.id='test-'+(v.id||'row-'+v.position);
 const head=el('div','test-card-head');
 head.append(el('span','test-state',c.state||section),el('span','test-age',c.age||'no time on record'),componentLink(v));
 a.append(head);
 a.append(...longText('p','test-q',c.question||v.title||'',testKey(v)+'|q',300));
 const facts=el('dl','test-facts');
 const fact=(label,text,cls,key)=>{const dd=el('dd',cls);dd.append(...longText('span','',text,testKey(v)+'|'+key,240));facts.append(el('dt','',label),dd);};
 fact('Direct target',c.direct_target||'Not written in the row: a row written before Oct 5, 2026 1:23 AM PDT does not owe it.',c.direct_target?'':'missing','t');
 fact('Conclusion',c.conclusion||'No conclusion is written.','','c');
 fact('Next action',c.next_action||'No next action is written in the row.','','n');
 a.append(facts);
 if(section!=='DONE')a.append(v.diagram?diagramStrip(v.diagram):el('p','test-no-diagram','No diagram is written in the row (listed under Errors).'));
 a.append(testDetails(v,open));
 // the movement viewer (movement_viewer.js): a button to the saved movements of the recordings this row names
 if(window.movementViewer)window.movementViewer.testButton(a,v.id);
 return a;
}
function openTestsSection(key){
 const block=document.querySelector('.test-section-block[data-section="'+key+'"]');if(!block)return null;
 block.open=true;testsSectionOpen.set(key,true);return block;
}
function showTestCard(row){
 const card=document.getElementById('test-'+row);if(!card)return;
 openTestsSection(card.dataset.section);card.scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
 card.classList.add('flash');setTimeout(()=>card.classList.remove('flash'),1600);
}
function testsSectionBlock(s,open){
 const block=el('details','test-section-block tone-'+(SECTION_TONE[s.key]||'blocked'));block.dataset.section=s.key;block.id='section-'+s.key;
 block.open=testsSectionOpen.has(s.key)?testsSectionOpen.get(s.key):Boolean(s.open_by_default);
 const sum=el('summary','test-section-summary');
 sum.append(el('span','section-name',s.key),el('span','section-count',String(s.count)),el('span','section-meaning',s.meaning||''));
 block.append(sum);
 block.addEventListener('toggle',()=>testsSectionOpen.set(s.key,block.open));
 const rows=s.rows||[];
 if(!rows.length)block.append(el('p','small muted section-empty','No row is in this section.'));
 for(const v of rows)block.append(testCard(v,open));
 return block;
}
function namesBlock(d){
 const body=$('tests-names-body');if(!body)return;body.replaceChildren();
 const vocab=el('dl','names-vocab'),comps=el('div','names-components');
 const add=(name,definition)=>{vocab.append(el('dt','',name),el('dd','',definition));};
 for(const s of d.sections||[])add(s.key,s.meaning||'');
 for(const n of d.names_used||[])if(n.kind!=='section'&&n.kind!=='component')add(n.name,n.definition);
 for(const [name,definition] of CARD_TERMS)add(name,definition);
 if((d.sections||[]).some(s=>(s.rows||[]).some(v=>(v.readings||[]).length)))for(const [name,definition] of JUDGEMENT_TERMS)add(name,definition);
 // the investigations section's own terms, built by the server (tools/elimination.py NAMES), when it shows a diagram
 if((invData?.investigations||[]).some(i=>!i.refused))for(const n of invData.names||[])add(n.name,n.definition);
 const named=(d.names_used||[]).filter(n=>n.kind==='component');
 body.append(vocab);
 if(named.length){
  body.append(el('h3','names-subhead','The components the rows name ('+named.length+')'));
  for(const n of named){
   const det=el('details','name-component'),sum=el('summary');const cut=String(n.definition||'').indexOf(': ');
   sum.append(el('code','',n.name),el('span','name-plain',cut>0?n.definition.slice(0,cut):''));det.append(sum,el('p','',cut>0?n.definition.slice(cut+2):n.definition));comps.append(det);
  }
  body.append(comps);
 }
}
function renderTestsErrors(d){
 const box=$('tests-errors'),list=$('tests-errors-list');if(!box)return;list.replaceChildren();
 const errors=d.errors||[],audit=d.audit||{},warnings=d.warnings||[];
 $('tests-errors-title').textContent=errors.length?plural(errors.length,'error')+' on this page':'No errors on this page';
 $('tests-audit').textContent=audit.sentence?'The audit of live test processes against the running rows (both directions): '+audit.sentence:'';
 box.classList.toggle('clear',!errors.length);
 for(const e of errors){
  const li=el('li');li.append(el('strong','',(e.kind||'error')+': '),document.createTextNode(e.text||''));
  if(e.row&&document.getElementById('test-'+e.row)){const b=el('button','text-button','Show the row');b.type='button';b.addEventListener('click',()=>showTestCard(e.row));li.append(' ',b);}
  list.append(li);
 }
 // the development runs the audit lists: no test of a row, each with its owner and purpose (finding F10)
 let warn=$('tests-warnings');if(warn)warn.remove();
 if(warnings.length){
  warn=el('div','tests-warnings');warn.id='tests-warnings';
  warn.append(el('h3','tests-warnings-title',plural(warnings.length,'warning')+': development runs without a registry row'));
  const ul=el('ul','tests-error-list warnings');
  for(const w of warnings){const li=el('li');li.append(el('strong','',(w.kind||'warning')+': '),document.createTextNode(w.text||''));ul.append(li);}
  warn.append(ul);list.after(warn);
 }
 box.hidden=!errors.length&&!audit.sentence&&!warnings.length;
}
function renderTests(){
 const box=$('tests-groups');if(!box)return;const d=testsData;
 if(!d){$('tests-status').textContent=testsError||'Reading the test registry.';return;}
 if(!d.available||!Array.isArray(d.sections)){
  $('tests-status').textContent=(d.available?RESTART_NOTE:d.sentence||d.error||RESTART_NOTE)+(testsError?' '+testsError:'');$('tests-status').classList.add('error');return;
 }
 $('tests-status').textContent=testsError;$('tests-status').classList.toggle('error',Boolean(testsError));$('tests-status').hidden=!testsError;
 const src=$('tests-source');src.replaceChildren(document.createTextNode('Read from '),Object.assign(el('a','',d.registry),{href:d.registry_url}),
  document.createTextNode(' at '+d.read_at_text+(d.modified_text?'; the file was last changed '+d.modified_text:'')+'. '+(d.note||'')));
 $('tests-tally').textContent=d.sentence||'';
 const gaps=$('tests-gaps');gaps.replaceChildren();
 if(d.scheduler_gaps?.length){const s=el('section','test-gaps');s.append(el('h2','',d.scheduler_gaps_title||'Scheduled ability checks without a complete row'));
  if(d.scheduler_gaps_note)s.append(el('p','small',d.scheduler_gaps_note));
  const ul=el('ul');for(const g of d.scheduler_gaps){const li=el('li');li.append(el('strong','',g.title+': '),document.createTextNode(g.text+' '),el('code','',g.id));ul.append(li);}
  s.append(ul);gaps.append(s);}
 const open=new Set([...box.querySelectorAll('details.test-details[open]')].map(x=>x.dataset.key));
 const y=window.scrollY;box.replaceChildren();
 for(const s of d.sections)box.append(testsSectionBlock(s,open));
 namesBlock(d);renderTestsErrors(d);
 for(const id of ['tests-groups','tests-errors-list','tests-warnings','tests-gaps','tests-names-body'])markAliases($(id));
 const jump=$('tests-jump');jump.replaceChildren();
 for(const s of d.sections){const b=el('button','jump tone-'+(SECTION_TONE[s.key]||'blocked'));b.type='button';b.append(el('span','',s.key),el('strong','',String(s.count)));
  b.addEventListener('click',()=>{const block=openTestsSection(s.key);if(block)block.scrollIntoView({block:'start'});});jump.append(b);}
 // ?expand opens every section and every row's details, as on the Development view (reading everything, or a full-page check)
 if(new URLSearchParams(location.search).get('expand')!==null)for(const x of box.querySelectorAll('details'))x.open=true;
 if(Math.abs(window.scrollY-y)>1)window.scrollTo(0,y);
}
setInterval(()=>{if(!$('tests').hidden)loadTests();},30000);
// at a phone's width the names would fill the first two screens: there they start closed under their heading
if(compact.matches&&$('tests-names'))$('tests-names').open=false;

// ---------------------------------------------------------------- the top of the dashboard
// GET /api/overview (tools/recovery_api.py overview) and GET /api/machine_mode (tools/machine_mode.py), read every
// 30 seconds on every tab (the recovery directive of Oct 5, 2026, sections 5, 10, 12 and 13):
//  - the machine mode now, exactly one of TRAINING, TEST WINDOW and PAUSED FOR OPERATOR, in the band under the tabs;
//  - the gain-of-function scoreboard: its headline and the five stages, each with its state and what holds it back;
//  - the open GOF-BLOCKER defects first in line (at most 5; the ones with no meaningful work for 12 hours or more come
//    first), each with its age, its idle time, its next action and its owner;
//  - every open defect as NAME: STATE, and the errors the server lists;
//  - the week's hours in each machine mode (TRAINING split into the production lineage's runs and the registered
//    experiment runs), and each training pause with its reason.
// Every sentence is the server's; the page lays them out and adds only labels.
let overviewData=null,weekData=null,overviewError='',overviewRequest=0;
const MODE_TONE={'TRAINING':'training','TEST WINDOW':'window','PAUSED FOR OPERATOR':'paused'};
async function loadOverview(){
 const mine=++overviewRequest;
 const get=async path=>{const r=await fetch(path,{cache:'no-store',signal:AbortSignal.timeout(30000)});if(r.status===404)throw Error('404');if(!r.ok)throw Error('the server answered '+r.status);return r.json();};
 try{const [o,m]=await Promise.all([get('/api/overview'),get('/api/machine_mode')]);if(mine!==overviewRequest)return;overviewData=o;weekData=m;overviewError='';}
 catch(e){if(mine!==overviewRequest)return;overviewError=e.message==='404'?RESTART_NOTE:'The top of the dashboard could not be read ('+e.message+'); the page tries again every 30 seconds.';}
 renderModeStrip();renderTop();
}
function renderModeStrip(){
 const strip=$('machine-mode');if(!strip)return;const m=overviewData?.machine_mode;
 if(!m||!m.mode){strip.className='mode-strip unknown';$('mode-label').textContent='Machine mode unknown';$('mode-sentence').textContent=overviewError||m?.sentence||'Reading the machine mode.';return;}
 strip.className='mode-strip '+(MODE_TONE[m.mode]||'unknown');strip.dataset.mode=m.mode;$('mode-label').textContent=m.mode;
 // the label already shows the mode: "TRAINING: the trainer..." reads "the trainer...", and "TRAINING (experiment run,
 // not production): the trainer..." reads "Experiment run, not production: the trainer..." (tools/machine_mode.py)
 const said=String(m.sentence||'').replace(m.mode+': ','').replace(new RegExp('^'+m.mode+' \\(([^)]*)\\): '),(_,k)=>k.charAt(0).toUpperCase()+k.slice(1)+': ');
 $('mode-sentence').textContent=said+(overviewError?' ('+overviewError+')':'');
 // where training is held and why, one line under the sentence that opens to the reason
 const held=m.mode!=='TRAINING'&&Number.isFinite(m.held_at_step),reason=m.mode==='TEST WINDOW'?m.reason:null;
 let more=$('mode-more');if(more)more.remove();
 if(held||reason){
  more=el('details','mode-more');more.id='mode-more';
  more.append(el('summary','',held?'Training is held at model step '+stepText(m.held_at_step)+(m.active_run?' of run '+m.active_run:''):'The reason on record'));
  if(reason)more.append(el('p','','Reason on record: '+String(reason).replace(/\.$/,'')+'.'));
  strip.append(more);
 }
 markAliases($('mode-sentence'));markAliases($('mode-more'));
}
// "13 days 4 hours since it was found (a day without its time of day)": the span alone ("13 days 4 hours") on the line,
// the whole sentence in the element's title and in the blocker's own details, so nothing is lost
function spanOf(text,cls){
 const s=String(text||''),m=s.match(/^((?:\d+ (?:days?|hours?|minutes?) ?){1,2}|under 1 minute)/);
 const n=el('span',cls||'',m?m[1].trim():s);n.title=s;return n;
}
// what an idle time counts from, in a few words beside it, the full sentence in their title (the operator, Oct 5,
// 2026: "last worked" means when substantive work happened, not when somebody edited the register)
function idleMeaning(b,o){
 const s=el('span','idle-meaning',b.idle_short?' ('+b.idle_short+')':'');
 if(o.defects?.idle_meaning)s.title='The idle time is '+o.defects.idle_meaning+'.';return s;
}
// a long canonical id may break after any underscore, never inside a word
function breakableId(text,cls){
 const n=el('span',cls||''),parts=String(text||'').split('_');
 parts.forEach((part,i)=>{n.append(document.createTextNode(part+(i<parts.length-1?'_':'')));if(i<parts.length-1)n.append(document.createElement('wbr'));});
 return n;
}
// hours with their unit right after the number: "46.1 hours", "0 hours", "1 hour"
const hoursText=h=>{const v=Math.round(h*10)/10;return (Number.isInteger(v)?String(v):v.toFixed(1))+(v===1?' hour':' hours');};
function renderTop(){
 const o=overviewData;
 if(!o){$('scoreboard-headline').textContent=overviewError||'Reading the scoreboard.';return;}
 const board=o.scoreboard||{},stages=board.stages||[];
 $('scoreboard-headline').textContent=board.sentence||board.headline||'';
 $('scoreboard-headline').classList.toggle('none',!board.gain_of_function_demonstrated);
 // a reason every stage carries is printed once, under the five stages
 const shared=stages.length?(stages[0].reasons||[]).filter(r=>stages.every(s=>(s.reasons||[]).includes(r))):[];
 const list=$('scoreboard-stages');list.replaceChildren();
 for(const s of stages){
  const li=el('li','stage '+(s.state==='DEMONSTRATED'?'shown':'open'));li.dataset.stage=s.key;
  if(s.required_evidence)li.title='Required evidence: '+s.required_evidence;
  li.append(el('span','stage-name',s.name),el('span','stage-state',s.state));
  const held=s.blocking_defects||[];
  // the blocking defects as NAME: STATE lines; the server's sentence that lists them is not printed a second time
  for(const r of (s.reasons||[]).filter(r=>!shared.includes(r)&&!(held.length&&/^\d+ open GOF-BLOCKER/.test(r))))li.append(el('p','stage-reason',r));
  if(held.length){
   const ul=el('ul','stage-held');ul.setAttribute('aria-label','Open GOF-BLOCKER defects that name this stage');
   for(const b of held){const item=el('li');item.append(breakableId(b.id+': '+b.state));ul.append(item);}
   li.append(el('span','stage-held-label','Held back by'),ul);
  }
  list.append(li);
 }
 const sharedLine=$('scoreboard-shared');sharedLine.replaceChildren();
 if(shared.length)sharedLine.append(el('strong','','Every stage: '),document.createTextNode(shared.join(' ')+' '));
 if(board.how_a_stage_is_shown){const d=el('details','inline-details');d.append(el('summary','','How a stage is shown'),el('p','',board.how_a_stage_is_shown));sharedLine.append(d);}
 // the GOF-BLOCKER defects first in line: on the card the id and state, the idle time, the age, the owner and the next
 // action; what the defect is and what it waits on in the card's own details
 const blockers=o.top_blockers||[],bl=$('top-blockers');bl.replaceChildren();
 $('blockers-sentence').textContent=o.defects?.sentence||'';$('blockers-sentence').title=o.defects?.rule||'';
 for(const b of blockers){
  const li=el('li','blocker'+(b.top?' idle':'')+(b.needs_explanation?' held':''));li.dataset.defect=b.id;
  const head=el('div','blocker-head');head.append(breakableId(b.id+': '+b.state,'blocker-status'));
  if(b.component)head.append(el('span','blocker-component',b.component));
  if(b.top)head.append(el('span','blocker-flag','idle 12 hours or more'));
  if(b.needs_explanation)head.append(el('span','blocker-flag hold','idle 24 hours: waits for the coordinator’s explanation'));
  li.append(head);
  const facts=el('p','blocker-facts');
  // what the idle time counts from, beside it (the operator, Oct 5, 2026: "last worked" is when the work was done)
  facts.append(el('span','fact-label','Idle'),spanOf(b.idle_text),idleMeaning(b,o),document.createTextNode(' · '),
   el('span','fact-label','Found'),spanOf(b.age_text),document.createTextNode(' ago · '),
   el('span','fact-label','Owner'),el('span','',b.owner||'not on record'));
  li.append(facts);
  const next=el('p','blocker-next');next.append(el('span','fact-label','Next action'),document.createTextNode(b.next_action||'not on record'));li.append(next);
  const more=el('details','blocker-more');more.append(el('summary','','What it is, what it waits on'));
  if(b.title)more.append(el('p','blocker-title',b.title));
  if(b.waits_on){const w=el('p','blocker-waits');w.append(el('span','fact-label','Waits on'),document.createTextNode(b.waits_on));more.append(w);}
  for(const t of [b.age_text,b.idle_text])if(t)more.append(el('p','blocker-waits',t.charAt(0).toUpperCase()+t.slice(1)+(t===b.idle_text&&o.defects?.idle_meaning?' (The idle time is '+o.defects.idle_meaning+'.)':'.')));
  li.append(more);
  bl.append(li);
 }
 if(!blockers.length)bl.append(el('li','blocker none','No open GOF-BLOCKER defect is on record.'));
 // every open defect as NAME: STATE
 const lines=o.status_lines||[],ul=$('status-lines-list');ul.replaceChildren();
 for(const s of lines)ul.append(el('li','status-chip',s));
 $('status-lines-summary').textContent=plural(lines.length,'open defect')+' as NAME: STATE';
 // the errors the server lists (the audit, the machine mode, the scoreboard, the defect register, the manifest's check)
 const errors=o.errors||[],eu=$('overview-errors-list');eu.replaceChildren();
 for(const e of errors){const li=el('li');li.append(el('strong','',(e.source||'error')+': '),document.createTextNode(e.text||''));eu.append(li);}
 $('overview-errors').classList.toggle('clear',!errors.length);
 $('overview-errors-summary').textContent=errors.length?plural(errors.length,'error')+' on record':'No errors on record';
 renderWeek();
 // the manifest check's findings are among the errors above; its sentence stays in this line's title
 $('gof-top-source').textContent='Read '+(o.read_at_text||'')+' from the defect register, the test registry and the machine’s processes.';
 $('gof-top-source').title=o.manifest?.sentence||'';
 for(const id of ['scoreboard-stages','scoreboard-shared','top-blockers','blockers-sentence','status-lines-list','overview-errors-list','week-pauses-list'])markAliases($(id));
}
function renderWeek(){
 const w=weekData?.week;if(!w){$('week-window').textContent=overviewError||'Reading the week.';return;}
 const total=w.window_hours||168,hours=w.hours||{};
 $('week-window').textContent='The last '+total+' hours: '+w.window_from+' to '+w.window_to+'.';
 const bar=$('week-bar'),legend=$('week-legend');bar.replaceChildren();legend.replaceChildren();
 // TRAINING carries its split: the production lineage's trainer lives and the registered experiment runs' lives
 // outside them ("TRAINING 19.01 hours (production 12.33, experiment 6.67)"); the experiment runs are in its title
 const split=w.training_split,shown=mode=>{const h=Number(hours[mode])||0;return mode==='TRAINING'&&split?h.toFixed(2)+' hours (production '+Number(split.production||0).toFixed(2)+', experiment '+Number(split.experiment||0).toFixed(2)+')':hoursText(h);};
 const runs=(w.experiment_runs||[]).map(e=>e.run+' (row '+e.row+', '+e.from_text+' to '+e.to_text+', '+Number(e.hours).toFixed(2)+' hours)');
 for(const [mode,tone] of Object.entries(MODE_TONE)){
  const h=Number(hours[mode])||0,share=Math.max(0,Math.min(100,100*h/total));
  const seg=el('span','week-seg '+tone);seg.style.width=share+'%';seg.title=mode+': '+shown(mode)+' of the '+total;bar.append(seg);
  const li=el('li','week-key '+tone);li.append(el('span','swatch '+tone),el('strong','',mode+' '),document.createTextNode(shown(mode)));
  if(mode==='TRAINING'&&runs.length)li.title='Experiment runs: '+runs.join('; ')+'.';
  legend.append(li);
 }
 bar.setAttribute('aria-label','Hours this week, of '+total+': '+Object.keys(MODE_TONE).map(m=>m+' '+shown(m)).join(', '));
 // each reason once, newest pause first, with every pause it explains
 const groups=new Map();
 for(const p of [...(w.pauses||[])].reverse()){const k=p.reason||'not recorded';if(!groups.has(k))groups.set(k,[]);groups.get(k).push(p);}
 const pl=$('week-pauses-list');pl.replaceChildren();
 for(const [reason,ps] of groups){
  const li=el('li');li.append(el('p','pause-reason',reason));
  const spans=el('ul','pause-spans');
  for(const p of ps)spans.append(el('li','',p.from_text+' to '+p.to_text+': '+Number(p.hours).toFixed(2)+' hours, of them '+Number(p.test_window_hours).toFixed(2)+' hours in TEST WINDOW'+(Number(p.experiment_training_hours)>0?' and '+Number(p.experiment_training_hours).toFixed(2)+' hours of experiment training, counted as TRAINING ('+(p.experiment_runs||[]).join(', ')+')':'')+(p.reason_source?' (reason read from '+p.reason_source+')':'')+(p.open?'; this pause is still open':'')));
  li.append(spans);pl.append(li);
 }
 $('week-pauses-summary').textContent=plural((w.pauses||[]).length,'training pause')+' this week, with '+plural(groups.size,'distinct reason');
 $('week-not-carried').textContent=(w.definition?w.definition+' ':'')+((w.not_carried||[]).length?'What the records do not carry: '+w.not_carried.join('; ')+'.':'');
}

// ---------------------------------------------------------------- the two live tabs
// The picture is one file the run replaces in place, so the page re-requests it with a
// changing query string; the browser would otherwise show the copy it already has.
let liveTimer=null,liveClip=null,liveBusy=false,liveRevision=0,lastLiveApiAt=0,liveReadPending=false,liveApiError=false;
function liveShown(){return !$('live').hidden;}
async function post(body){
 const response=await fetch('/api/live',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(body.action==='stop'||body.action==='save'?120000:15000)});
 const result=await response.json();if(!response.ok||result.error)throw Error(result.error||'Server returned '+response.status);return result;
}
function liveFresh(kind,now=Date.now()/1000){
 const l=state?.live||{},elapsed=Math.max(0,now-(l.observed_at||state?.generated_at||now));
 return !liveApiError&&now-(lastLiveApiAt||lastApiAt)<=75&&l[kind+'_publishing']===true&&Number.isFinite(l[kind+'_age_seconds'])&&l[kind+'_age_seconds']+elapsed<=(l[kind+'_fresh_seconds']||15);
}
function updateLiveFreshness(now=Date.now()/1000){
 if(!state)return;const l=state.live||{},frame=liveFresh('frame',now),pose=liveFresh('state',now);
 $('live-missing').hidden=frame;$('live-body').hidden=!frame;
 const paused=state.verdict?.state==='paused';
 $('live-missing').textContent=paused?'Training is paused. The live camera will update after training resumes; saved clips are under Video evidence.':'No current camera picture. The last image is missing or overdue; check publishing settings in Training details.';
 // the experiment view's picture is a copy: its trainer reads no recording or viewer request (POST /api/live refuses)
 const copy=l.view==='experiment';
 $('live-record').disabled=liveBusy||copy||(!frame&&!l.recording);
 $('open-viewer').disabled=liveBusy||copy||!pose;$('sensors-toggle').disabled=liveBusy||copy||!pose||!l.sensors_published;
 $('viewer-availability').textContent=copy?'The 3D viewer follows production runs only: the experiment mirror copies the experiment run’s picture, not its body state.':pose?'Current body state available.':paused?'Training is paused. The 3D viewer needs fresh body updates.':'No current body state. The 3D viewer cannot follow the run.';
 const seconds=Number.isFinite(l.frame_age_seconds)?l.frame_age_seconds+Math.max(0,now-(l.observed_at||now)):null;
 $('live-age').textContent=seconds===null?'No picture':'Picture recorded '+age(seconds)+' ago';
 $('live-fresh').textContent=seconds===null?'No picture':age(seconds)+' ago';
 $('live-pub').textContent=frame?'Current picture available':'No current picture';
}
function paintLive(){
 const l=state?.live||{},tx=l.texts||{};
 $('live-kept').textContent=l.kept_frames||0;
 if(!liveBusy){$('live-status').textContent=l.view==='experiment'?'Recording unavailable: this picture is a copy of the experiment run '+l.run_name+'’s live picture':l.recording?('Recording: '+(l.kept_frames||0)+' frames kept'):'Not recording';$('live-record').textContent=l.recording?'Stop and save':'Record';}
 $('sensors-toggle').textContent=l.show_sensors?'Hide sensors':'Show sensors';
 if(l.viewer_command)$('viewer-cmd').textContent=l.viewer_command;
 $('live-touch-summary').textContent=tx.touch_summary||'No current touch count.';
 $('live-selftouch').textContent=tx.selftouch||'Unmeasured';$('live-selfpred').textContent=tx.selfpred||'No self-touch scored yet';
 $('live-posture').textContent=tx.posture||'Unmeasured';
 const regions=$('live-regions');regions.replaceChildren();for(const row of (tx.regions||'').split(' | ').filter(Boolean))regions.append(el('p','',row));
 const ready=$('live-ready');ready.replaceChildren();for(const clip of l.ready||[]){const a=el('a','',clip.name);a.href=clip.url;ready.append(a);}
 updateLiveFreshness();if(l.frame&&liveShown()&&liveFresh('frame'))$('live-frame').src=l.frame+'?t='+Date.now();
}
async function liveTick(){
 if(!state||(!liveShown()&&$('viewer').hidden)||liveReadPending||liveBusy)return;
 liveReadPending=true;const revision=liveRevision;
 try{const r=await fetch('/api/live',{cache:'no-store',signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error('Live status unavailable');const next=await r.json();
  if(revision===liveRevision&&!liveBusy){state.live=next;lastLiveApiAt=Date.now()/1000;liveApiError=false;paintLive();}
 }catch(e){liveApiError=true;updateLiveFreshness();}finally{liveReadPending=false;}
}
async function liveAction(button,status,operation){
 if(liveBusy)return;liveBusy=true;liveRevision++;button.disabled=true;updateLiveFreshness();
 try{await operation();}catch(e){status.textContent='Action could not be confirmed: '+e.message+'. Check the status before retrying.';}
 finally{liveBusy=false;liveRevision++;updateLiveFreshness();}
}
$('live-record').addEventListener('click',()=>liveAction($('live-record'),$('live-action-result'),async()=>{
 $('live-action-result').textContent='';const l=state?.live;if(!l)throw Error('Current recording state unavailable');
 if(!l.recording){const r=await post({action:'record'});if(r.recording!==true)throw Error('Recording was not confirmed');Object.assign(state.live,r);paintLive();$('live-action-result').textContent='Recording: 0 frames kept';$('live-record').textContent='Stop and save';return;}
 const r=await post({action:'stop'});Object.assign(state.live,r);$('live-record').textContent='Record';
 if(!r.clip){$('live-action-result').textContent='Stopped. No frames were captured.';return;}
 liveClip=r.clip;const name=prompt('Name this recording ('+r.clip.frames+' frames):','');
 if(!name){$('live-action-result').textContent='Recording kept as '+r.clip.file;const a=el('a','',r.clip.file);a.href='/files/'+state.training.run+'/live/rec/'+r.clip.file;$('live-ready').append(a);return;}
 const saved=await post({action:'save',file:r.clip.file,name:name});$('live-action-result').textContent='Saved as '+saved.name;
 const a=el('a','',saved.name);a.href='/files/'+saved.saved;$('live-ready').append(a);
}));
$('open-viewer').addEventListener('click',()=>liveAction($('open-viewer'),$('viewer-status'),async()=>{
 $('viewer-status').textContent='Opening the viewer...';const r=await post({action:'open_viewer'});
 if(!r.opened)throw Error('Viewer launch was not confirmed');$('viewer-status').textContent='Viewer launched. Its window may take a few seconds to appear.';
}));
$('sensors-toggle').addEventListener('click',()=>liveAction($('sensors-toggle'),$('viewer-status'),async()=>{
 const r=await post({action:'sensors',show:!state?.live?.show_sensors});if(typeof r.show_sensors!=='boolean')throw Error('Sensor setting was not confirmed');
 state.live.show_sensors=r.show_sensors;$('sensors-toggle').textContent=r.show_sensors?'Hide sensors':'Show sensors';
 $('viewer-status').textContent=r.show_sensors?'Sensor areas shown in the viewer.':'Sensor areas hidden in the viewer.';
}));
liveTimer=setInterval(liveTick,1500);
refresh();setInterval(refresh,30000);setInterval(updateClocks,1000);
