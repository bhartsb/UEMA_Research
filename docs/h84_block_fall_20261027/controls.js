(() => {
const time=document.getElementById('block-time'),play=document.getElementById('block-play');
let running=false,start=0,base=0,raf=0;
const cache=new Map();
function prepared(k){
 if(!cache.has(k)){const pair=['side','front'].map(view=>{const im=new Image();im.src='h84_block_fall_20261027/knee_contact/block_'+view+'_'+String(k).padStart(3,'0')+'.webp';return im});cache.set(k,pair);}
 return cache.get(k);
}
function warm(k){for(let n=k;n<=Math.min(360,k+12);n++)prepared(n);for(const n of cache.keys())if(n<k-2||n>k+12)cache.delete(n);}
const phases=[
[.65,'Upright on the 9-inch block. The feet support the body.','Standing on the block.'],
[1.6,'The trunk tips toward the edge as the arms move forward. The heels rise while the toes stay near the edge.','Support shifts toward the toes.'],
[1.86,'The pelvis descends; hips and knees bend to clear the block, while the arms approach the floor.','Descending toward the floor.'],
[3.9,'Both knees meet the floor and stay down. The elbows bend as the chest lowers, then extend as the chest rises.','Supported on hands and knees.'],
[5.5,'The hips shift back toward the heels before the hands lift; the chest then rises over the knees. The arms return beside the trunk.','Moving support from hands to knees.'],
[7.3,'The right foot swings forward and plants. The left knee remains down while the trunk shifts toward the right foot.','Kneeling, then planting one foot.'],
[10.3,'The left knee lifts and its foot comes forward. Both hips and knees extend over the planted feet.','Rising over the feet.'],
[13,'Upright again; the final stance is held for inspection while the example touch-precision state continues recovering.','Standing on the floor.']];
function show(k){k=Math.max(0,Math.min(360,Math.round(k)));warm(k);const pair=prepared(k);if(running&&!pair.every(im=>im.complete&&im.naturalWidth))return;time.value=k;const t=k/30;
document.getElementById('block-clock').textContent=t.toFixed(1)+' / 12.0 s';
for(const [index,view] of ['side','front'].entries()){let im=document.getElementById('block-'+view);if(running){const ready=pair[index];ready.id=im.id;if(ready!==im)im.replaceWith(ready);im=ready;}else{im.src='h84_block_fall_20261027/knee_contact/block_'+view+'_'+String(k).padStart(3,'0')+'.webp';}im.alt='MIMo block fall and get-up, '+view+' view at '+t.toFixed(1)+' seconds';}
const phase=phases.find(p=>t<p[0]);document.getElementById('block-caption').textContent=phase[1];document.getElementById('block-body').textContent=phase[2];
const q=t<1.86?1:1-.8*Math.exp(-(t-1.86)/10),bits=Math.round(2+21*q),step=100000/(2**bits-1);
document.getElementById('block-meter').value=q;
document.getElementById('block-quality').textContent=t<1.86?'Healthy example: readings pass through unchanged.':'Illustrative local q = '+q.toFixed(2)+': '+bits+'-bit pressure steps, about '+step.toFixed(2)+' Pa apart. This is not a percentage of accuracy or strength.';
}
function stop(){running=false;cancelAnimationFrame(raf);raf=0;play.textContent='Play fall and get-up';}
function tick(now){raf=0;if(!running)return;const k=base+(now-start)*30/1000;show(k);if(k>=360)stop();else raf=requestAnimationFrame(tick);}
play.onclick=()=>{if(running){stop();return;}if(+time.value>=360)show(0);base=+time.value;start=performance.now();running=true;play.textContent='Pause';raf=requestAnimationFrame(tick);};
time.oninput=()=>{stop();cache.clear();show(+time.value);};document.getElementById('block-reset').onclick=()=>{stop();cache.clear();show(0);};
document.getElementById('block-view').onchange=e=>{for(const v of ['side','front'])document.getElementById('block-'+v+'-figure').hidden=e.target.value!=='both'&&e.target.value!==v;document.getElementById('block-viewer').style.gridTemplateColumns=e.target.value==='both'?'':'1fr';};
show(0);
})();
