(()=>{
'use strict';
const D=window.DEMO_DATA,S=D.scene,R=window.RI,$=id=>document.getElementById(id);
let logits,removed,trail,position,oldPath,newPath,selected,corrected,playing=false,timer,iso=true,invalidated=[],forecastCache=new Map(),revisionPosition;
const w=S.width,h=S.height,canvas=$('world'),ctx=canvas.getContext('2d');
const pct=n=>(100*n).toFixed(1)+'%';
function toast(message){$('toast').textContent=message;$('toast').classList.add('visible');setTimeout(()=>$('toast').classList.remove('visible'),2800);}
function reset(){clearInterval(timer);playing=false;removed=new Set();logits=R.infer(S);trail=Array.from({length:6},(_,i)=>4*w+3+i);position=trail.at(-1);oldPath=R.plan(logits,w,h,position,S.goal);newPath=oldPath.slice();selected='E006';corrected=false;invalidated=[];forecastCache=new Map();$('retract').disabled=false;$('play').textContent='▶ Execute plan';update();}
function routes(){
 const at=position,gy=Math.floor(S.goal/w),gx=S.goal%w;
 function via(y){const p=[];let x=at%w,cy=Math.floor(at/w);p.push(at);while(cy!==y){cy+=Math.sign(y-cy);p.push(cy*w+x);}while(x!==gx){x+=Math.sign(gx-x);p.push(cy*w+x);}while(cy!==gy){cy+=Math.sign(gy-cy);p.push(cy*w+x);}return p;}
 return [{name:'Upper passage',id:'shortcut',path:via(4),description:'Shortest candidate; depends on E006’s spatial factor.'},{name:'Lower passage',id:'detour',path:via(13),description:'Longer candidate supported by independent evidence.'},{name:'Hold position',id:'hold',path:[at],description:'Preserves the current physical state; does not reach the goal.'}];
}
function update(){
 const occ=R.sigmoid(logits[S.bad_cell]);$('occupancy').textContent=pct(occ);$('occupancy-meter').style.width=pct(occ);$('occupancy-note').textContent=corrected?'The unsupported shortcut is conservatively blocked.':'Evidence E006 dominates the prior.';
 $('phase-tag').textContent=corrected?'AFTER RETRACTION':'BEFORE RETRACTION';$('phase-tag').className='tag '+(corrected?'green':'amber');$('state-label').textContent=corrected?'REPAIRED':'UNCORRECTED';$('state-label').className='tag '+(corrected?'green':'amber');
 $('trail-label').textContent=(trail.length-1)+' actual actions retained · never erased';
 $('evidence-list').innerHTML=S.events.map(e=>`<button class="evidence-row ${e.id===selected?'selected':''} ${e.valid?'':'suspect'} ${removed.has(e.id)?'revoked':''}" data-id="${e.id}"><span class="dot"></span><span class="event-id">${e.id}</span><span class="source">${e.source}</span><span>${removed.has(e.id)?'×':e.valid?'✓':'!'}</span></button>`).join('');
 for(const b of $('evidence-list').querySelectorAll('button'))b.onclick=()=>{selected=b.dataset.id;update();};
 const e=S.events.find(e=>e.id===selected);$('packet-detail').innerHTML=`source &nbsp;<b>${e.source}</b><br>factor &nbsp;<b>cell (${e.cell%w}, ${Math.floor(e.cell/w)})</b><br>score &nbsp; <b>${e.delta.toFixed(3)} log-odds</b><br>status &nbsp;<b>${removed.has(e.id)?'RETRACTED':e.valid?'SUPPORTED':'AUDIT: INVALID'}</b>`;
 const ref=R.infer(S,removed),err=Math.max(...ref.map((v,i)=>Math.abs(v-logits[i])));
 $('agreement').textContent=corrected?err.toExponential(1):'Not yet tested';
 $('retention').textContent=corrected?'503 / 504':'Awaiting revision';
 $('cache-result').textContent=corrected?invalidated.length+' / 3 invalidated':'3 cached candidates';
 $('cache-note').textContent=corrected?(position===revisionPosition?'2 retained at revision · planning reruns globally':'Revision audit above; moving refreshes route forecasts.'):'Candidate selection reruns globally.';
 $('forecast-cards').innerHTML=routes().map(route=>{const key=route.path.join(',');let cached=forecastCache.get(route.id);if(!cached||cached.key!==key){cached={key,value:R.forecast(logits,route.path)};forecastCache.set(route.id,cached);}const f=cached.value,affected=invalidated.includes(route.id),status=corrected?(position===revisionPosition?(affected?'RECOMPUTED':'RETAINED'):'CURRENT-STATE FORECAST'):'CACHED FORECAST';return `<article class="forecast-card ${affected?'changed':''}"><span class="tag ${affected?'green':''}">${status}</span><h3>${route.name}</h3><div class="forecast-numbers"><div><span>P(no occupied cell)</span><strong>${pct(f.survival)}</strong></div><div><span>Steps</span><strong>${f.steps}</strong></div></div><p>${route.description}</p></article>`;}).join('');
 draw();
}
function retract(){
 if(corrected)return;
 const paths=routes();const t=performance.now();const result=R.retract(logits,S.events,S.bad_ids,removed);logits=result.logits;
 invalidated=paths.filter(r=>r.path.some(c=>result.cells.includes(c))).map(r=>r.id);for(const id of invalidated)forecastCache.delete(id);
 newPath=R.plan(logits,w,h,position,S.goal);corrected=true;revisionPosition=position;$('retract').disabled=true;
 $('world-caption').textContent='The state is repaired. Actual past actions remain; the route is replanned.';
 update();toast('E006 retracted · '+result.cells.length+' factor repaired · '+invalidated.length+' forecasts invalidated');
}
function step(){
 if(position===S.goal){stop();toast('Goal reached. The revised route avoided the real obstacle.');return;}
 const p=R.plan(logits,w,h,position,S.goal);if(p.length<2){stop();toast('No feasible route under the current belief.');return;}
 const next=p[1];if(S.truth[next]){stop();$('world-caption').textContent='Physical contact: the believed-open passage is blocked. No motion was executed.';toast('Collision prevented by the simulator. Retract E006 to repair the belief.');return;}
 position=next;trail.push(next);newPath=R.plan(logits,w,h,position,S.goal);update();
 if(position===S.goal){stop();toast('Goal reached through the verified lower passage.');}
}
function stop(){clearInterval(timer);playing=false;$('play').textContent='▶ Execute plan';}
function draw(){
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,W=rect.width,H=rect.height;
 if(canvas.width!==Math.round(W*dpr)||canvas.height!==Math.round(H*dpr)){canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
 const scale=Math.min((W-45)/(iso?(w+h)*.84:w),(H-80)/(iso?(w+h)*.43:h));
 const origin=iso?{x:W/2-(w-h)*scale*.42,y:35}:{x:(W-w*scale)/2,y:25};
 function project(x,y,z=0){return iso?[origin.x+(x-y)*scale*.84,origin.y+(x+y)*scale*.43-z*scale]:[origin.x+x*scale,origin.y+y*scale-z*scale*.5];}
 function poly(points,fill,stroke){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.6;ctx.stroke();}}
 const reveal=$('reveal').checked;
 function tile(x,y){const c=y*w+x,p=R.sigmoid(logits[c]),wall=p>.55,actual=S.truth[c];let fill=wall?'#344251':'#202c39';
 if(c===S.bad_cell)fill=corrected?'#536355':'#534561';
 if(reveal&&actual&&!wall)fill='#9a5f45';
 const base=[project(x,y),project(x+1,y),project(x+1,y+1),project(x,y+1)];poly(base,fill,'#344354');
 if(wall&&iso){const z=.65,a=project(x,y,z),b=project(x+1,y,z),d=project(x+1,y+1,z),e=project(x,y+1,z);poly([base[1],base[2],d,b],'#26313f');poly([base[2],base[3],e,d],'#2c3949');poly([a,b,d,e],c===S.bad_cell?'#819880':'#425268','#576779');}
 }
 if(iso){for(let sum=0;sum<w+h;sum++)for(let y=0;y<h;y++){const x=sum-y;if(x>=0&&x<w)tile(x,y);}}
 else for(let y=0;y<h;y++)for(let x=0;x<w;x++)tile(x,y);
 function line(path,color,dashed=false,width=2){if(!path.length)return;ctx.save();ctx.beginPath();for(let i=0;i<path.length;i++){const c=path[i],p=project(c%w+.5,Math.floor(c/w)+.5,.12);if(i)ctx.lineTo(...p);else ctx.moveTo(...p);}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[5,5]:[]);ctx.lineJoin='round';ctx.stroke();ctx.restore();}
 line(oldPath,'#b4a2ff',true,2);if(corrected)line(newPath,'#80e5cc',false,2.7);line(trail,'#eef2f8',false,2.5);
 function marker(c,color,label){const p=project(c%w+.5,Math.floor(c/w)+.5,.3);ctx.beginPath();ctx.arc(...p,label==='AGENT'?5.5:4,0,Math.PI*2);ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=10;ctx.fill();ctx.shadowBlur=0;ctx.fillStyle=color;ctx.font='8px Segoe UI';ctx.fillText(label,p[0]+9,p[1]-7);}
 marker(S.goal,'#80e5cc','GOAL');marker(position,'#fff','AGENT');
 const bad=project(14.5,4.5,.9);ctx.strokeStyle=corrected?'#80e5cc':'#f2c981';ctx.beginPath();ctx.arc(...bad,9,0,Math.PI*2);ctx.stroke();ctx.fillStyle=corrected?'#80e5cc':'#f2c981';ctx.font='9px Consolas';ctx.fillText('E006',bad[0]+13,bad[1]);
 const selectedEvent=S.events.find(e=>e.id===selected);if(selectedEvent&&selectedEvent.id!=='E006'){const p=project(selectedEvent.cell%w+.5,Math.floor(selectedEvent.cell/w)+.5,.3);ctx.strokeStyle='#b4a2ff';ctx.strokeRect(p[0]-5,p[1]-5,10,10);}
 ctx.fillStyle='#6c7b90';ctx.font='8px Consolas';ctx.fillText(reveal?'EVALUATOR TRUTH OVERLAY · NOT AGENT INPUT':'AGENT BELIEF · STATIC OCCUPANCY MODEL',16,18);
}
function results(){
 const ev=D.evaluation,study=window.STUDY_DATA,row=study.api_timing.rows.find(x=>x.length===32768),max=row.methods.full_replay.median_ms;
 const names={full_replay:'Full replay',checkpoint_replay:'Checkpoint replay',explicit_subtract:'Explicit subtraction',indexed_retract:'Evidence-indexed repair'};
 $('timing-chart').innerHTML=Object.entries(names).map(([id,name])=>`<div class="chart-row"><div class="chart-label"><span>${name}</span><strong>${row.methods[id].median_ms.toFixed(2)} ms</strong></div><div class="bar-track"><div class="bar" style="width:${100*row.methods[id].median_ms/max}%"></div></div></div>`).join('');
 $('training-time').textContent=pct(ev.training.test_accuracy);$('training-accuracy').textContent=ev.training.test_examples.toLocaleString()+' synthetic held-out samples. Brier score '+ev.training.test_brier.toFixed(4)+'.';
 const families=Object.values(study.tasks.families),total=study.tasks.total,successful=families.reduce((s,f)=>s+f.methods.indexed_retract.successes,0);
 $('task-result').textContent=successful+' / '+total;$('task-detail').textContent='Four controlled task families with randomized geometry. All exact correction baselines match. False-obstacle median route: 39 → 23 steps. Not a deployment estimate.';
}
$('retract').onclick=retract;$('reset').onclick=()=>{reset();$('world-caption').textContent='The believed-open shortcut crosses the upper passage.';};$('step').onclick=step;$('play').onclick=()=>{if(playing)stop();else{playing=true;$('play').textContent='Ⅱ Pause';timer=setInterval(step,180);}};
$('reveal').onchange=draw;$('view-mode').onclick=()=>{iso=!iso;$('view-mode').textContent=iso?'ISOMETRIC':'TOP-DOWN';draw();};
new ResizeObserver(draw).observe(canvas.parentElement);results();reset();
window.RI_DEMO={reset,retract,step,getState:()=>({corrected,position,trail:trail.slice(),removed:[...removed],invalidated:invalidated.slice(),logits:logits.slice()})};
})();
