(function(root){
  'use strict';
  const sigmoid=x=>1/(1+Math.exp(-Math.max(-40,Math.min(40,x))));
  function infer(scene,removed=new Set()){
    const logits=scene.prior.slice();
    for(const e of scene.events) if(!removed.has(e.id)) logits[e.cell]+=e.delta;
    return logits;
  }
  function retract(logits,events,ids,removed){
    const index=new Map(events.map(e=>[e.id,e]));
    for(const id of ids) if(!index.has(id)) throw Error('Unknown evidence ID');
    const next=logits.slice(), cells=new Set();
    for(const id of new Set(ids)) if(!removed.has(id)){
      const e=index.get(id);next[e.cell]-=e.delta;removed.add(id);cells.add(e.cell);
    }
    return {logits:next,cells:[...cells]};
  }
  function plan(logits,w,h,start,goal){
    const p=logits.map(sigmoid),cost=new Map([[start,0]]),previous=new Map();
    const queue=[[0,start]],closed=new Set();
    while(queue.length){
      queue.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
      const [,c]=queue.shift();if(closed.has(c))continue;closed.add(c);
      if(c===goal){const path=[c];let q=c;while(previous.has(q)){q=previous.get(q);path.push(q);}return path.reverse();}
      const x=c%w,y=Math.floor(c/w);
      for(const [nx,ny]of[[x+1,y],[x-1,y],[x,y+1],[x,y-1]]){
        if(nx<0||nx>=w||ny<0||ny>=h)continue;
        const n=ny*w+nx;if(p[n]>=.55&&n!==goal)continue;
        const g=cost.get(c)+1+3*-Math.log(Math.max(1-p[n],1e-12));
        if(g<(cost.get(n)??Infinity)){cost.set(n,g);previous.set(n,c);queue.push([g+Math.abs(nx-goal%w)+Math.abs(ny-Math.floor(goal/w)),n]);}
      }
    }
    return [];
  }
  function forecast(logits,path){
    const risk=[...new Set(path)].reduce((s,c)=>s-Math.log(Math.max(1-sigmoid(logits[c]),1e-12)),0);
    return {survival:Math.exp(-risk),riskCost:risk,steps:Math.max(0,path.length-1)};
  }
  const api={sigmoid,infer,retract,plan,forecast};
  if(typeof module!=='undefined')module.exports=api;
  else root.RI=api;
})(typeof window!=='undefined'?window:globalThis);
