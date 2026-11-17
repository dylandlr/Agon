// Independent re-implementation of the Unity job loop, to check the integration tests' claims.
const SQRT2=1.41421356, SPEED=0.34;
const H=(a,b)=>{const dx=Math.abs(a[0]-b[0]),dy=Math.abs(a[1]-b[1]),mn=Math.min(dx,dy);return (dx+dy-2*mn)+mn*SQRT2;};
function mkMap(w,h){ const t=new Array(w*h).fill('soil'); const walls=new Set();
  return {w,h,t,walls, at:(x,y)=>t[y*w+x],
    walkable(x,y){ if(x<0||y<0||x>=w||y>=h)return false; if(walls.has(x+','+y))return false;
      const k=t[y*w+x]; return k!=='rock'&&k!=='water'; },
    cost(x,y){ const k=t[y*w+x]; return k==='forest'?1.6:k==='sand'?1.3:k==='snow'?1.5:k==='floor'?0.8:1; }}; }
const N=[[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]];
function astar(m,s,d){ if(!m.walkable(d[0],d[1]))return null; if(s[0]===d[0]&&s[1]===d[1])return[];
  const key=(x,y)=>y*m.w+x,g=new Map(),par=new Map(),cl=new Set(),open=[[H(s,d),key(s[0],s[1])]];
  g.set(key(s[0],s[1]),0);
  while(open.length){ open.sort((a,b)=>a[0]-b[0]); const [,cur]=open.shift(); if(cl.has(cur))continue; cl.add(cur);
    const cx=cur%m.w,cy=(cur-cx)/m.w;
    if(cx===d[0]&&cy===d[1]){const p=[];let a=cur;while(par.has(a)){p.push([a%m.w,(a-a%m.w)/m.w]);a=par.get(a);}return p.reverse();}
    for(const[ox,oy]of N){const nx=cx+ox,ny=cy+oy; if(!m.walkable(nx,ny))continue;
      const dg=ox!==0&&oy!==0; if(dg&&(!m.walkable(cx+ox,cy)||!m.walkable(cx,cy+oy)))continue;
      const nk=key(nx,ny); if(cl.has(nk))continue;
      const t=g.get(cur)+(dg?SQRT2:1)*m.cost(nx,ny);
      if(t<(g.has(nk)?g.get(nk):Infinity)){g.set(nk,t);par.set(nk,cur);open.push([t+H([nx,ny],d),nk]);}}}
  return null;}

function mkPawn(m,x,y){ return {m,pos:[x,y],alive:true,asleep:false,hp:1,comfort:0,
  sat:1,hyd:1,ene:1, path:null,pi:0,prog:0, job:null,arrived:false,worked:0,done:0,failed:null}; }
const TH={food:0.40,water:0.45,rest:0.30}, FALL={food:0.006,water:0.009,rest:0.0075}, REST_REC=0.055;
function needsTick(p){
  const wear = p.comfort<0 ? 1+Math.min(1,-p.comfort*0.04) : 1;
  p.sat=Math.max(0,Math.min(1,p.sat-FALL.food*wear));
  p.hyd=Math.max(0,Math.min(1,p.hyd-FALL.water));
  p.ene=p.asleep?Math.min(1,p.ene+REST_REC):Math.max(0,p.ene-FALL.rest);
}
function mostUrgent(p){ let w=null,wm=0;
  for(const [k,v] of [['food',p.sat],['water',p.hyd],['rest',p.ene]]){
    if(v>=TH[k])continue; const m=TH[k]-v; if(!w||m>wm){w=k;wm=m;} } return w; }
function giveJob(p,things){
  const u=mostUrgent(p);
  const tryNeed=(k)=>{
    if(k==='food'){ const t=nearest(things,'FoodStore',p); return t?{def:'Eat',cell:t.pos,thing:t}:null; }
    if(k==='water'){ const t=nearest(things,'WaterSource',p); if(t)return {def:'Drink',cell:t.pos,thing:t};
      const e=waterEdge(p.m,p.pos); return e?{def:'Drink',cell:e,thing:null}:null; }
    if(k==='rest'){ const t=nearest(things,'Bed',p); return t?{def:'Sleep',cell:t.pos,thing:t}:{def:'Sleep',cell:p.pos,thing:null}; }
    return null;};
  if(u){ let j=tryNeed(u); if(j){claim(j,p);return j;}
    for(const k of ['food','water','rest']){ if(k===u)continue; const v=k==='food'?p.sat:k==='water'?p.hyd:p.ene;
      if(v>=TH[k])continue; j=tryNeed(k); if(j){claim(j,p);return j;} } }
  return {def:'Wait',cell:p.pos,thing:null,dur:10};
}
const claim=(j,p)=>{ if(j.thing)j.thing.res=p; };
function nearest(things,kind,p){ let b=null,bd=Infinity;
  for(const t of things){ if(t.kind!==kind)continue; if(t.stock>=0&&t.stock<=0)continue;
    if(t.res&&t.res!==p)continue; const d=H(p.pos,t.pos); if(d<bd){bd=d;b=t;} } return b; }
function waterEdge(m,from){ let b=null,bd=Infinity;
  for(let y=0;y<m.h;y++)for(let x=0;x<m.w;x++){ if(m.t[y*m.w+x]!=='water')continue;
    for(const[dx,dy]of [[0,1],[1,0],[0,-1],[-1,0]]){ const sx=x+dx,sy=y+dy;
      if(!m.walkable(sx,sy))continue; const d=H(from,[sx,sy]); if(d<bd){bd=d;b=[sx,sy];} } } return b; }
function startPath(p,dest){ p.path=null;p.pi=0;p.prog=0;
  if(dest[0]===p.pos[0]&&dest[1]===p.pos[1])return true;
  const r=astar(p.m,p.pos,dest); if(!r)return false; p.path=r; return r.length>0; }
function pathTick(p){ if(!p.path||p.pi>=p.path.length)return;
  const nx=p.path[p.pi]; if(!p.m.walkable(nx[0],nx[1])){ const d=p.path[p.path.length-1]; if(!startPath(p,d))p.path=null; return; }
  const dg=nx[0]!==p.pos[0]&&nx[1]!==p.pos[1];
  const c=(dg?SQRT2:1)*p.m.cost(nx[0],nx[1]);
  p.prog+=SPEED; if(p.prog<c)return; p.prog-=c; p.pos=nx; p.pi++;
  if(p.pi>=p.path.length){p.path=null;p.pi=0;p.prog=0;} }
const WORK={Eat:3,Drink:2,Sleep:40,Wait:10,Goto:1};
function jobsTick(p,things){
  if(!p.job){ p.job=giveJob(p,things); p.arrived=false; p.worked=0; }
  let st='Ongoing';
  const j=p.job;
  if(j.thing && j.thing.stock>=0 && j.thing.stock<=0) st='Failed';
  else if(!p.arrived){
    if(H(p.pos,j.cell)<=SQRT2+0.01){ p.arrived=true; p.path=null; }
    else { if(!p.path && !startPath(p,j.cell)) st='Failed';
           else { pathTick(p); st='Ongoing'; } }
  }
  if(st==='Ongoing' && p.arrived){
    if(j.def==='Sleep'){ p.asleep=true; if(p.ene>=0.98) st='Done'; }
    if(st==='Ongoing' && ++p.worked>=WORK[j.def]) st='Done';
  }
  if(st==='Ongoing')return;
  if(st==='Done'){
    if(j.def==='Eat'){ const take=j.thing.stock<0?0.65:Math.min(0.65,j.thing.stock); if(j.thing.stock>0)j.thing.stock-=take; p.sat=Math.min(1,p.sat+take); }
    if(j.def==='Drink'){ let take=0.8; if(j.thing&&j.thing.stock>=0){take=Math.min(take,j.thing.stock);j.thing.stock-=take;} p.hyd=Math.min(1,p.hyd+take); }
    if(j.def==='Sleep'){ p.asleep=false; }
    p.done++;
  } else p.failed=j;
  for(const t of things) if(t.res===p) t.res=null;
  p.path=null;p.asleep=false;p.job=null;
  p.job=giveJob(p,things); p.arrived=false; p.worked=0;
}
function healthTick(p){ let d=0;
  if(p.sat<=0)d+=0.004; if(p.hyd<=0)d+=0.007; if(p.ene<=0)d+=0.002;
  if(Math.abs(p.comfort)>12)d+=(Math.abs(p.comfort)-12)*0.0004;
  if(d>0){p.hp=Math.max(0,p.hp-d); if(p.hp<=0)p.alive=false;} else if(p.hp<1)p.hp=Math.min(1,p.hp+0.0015); }
function tick(p,things){ if(!p.alive)return; needsTick(p); jobsTick(p,things); healthTick(p);
  if(!p.alive){ for(const t of things) if(t.res===p)t.res=null; } }

let pass=0,fail=0; const T=(n,c,extra)=>{ if(c)pass++; else {fail++;console.log('FAIL: '+n+(extra!==undefined?'  ['+extra+']':''));} };

// 1. hungry pawn walks to store and eats, within 200 ticks
{ const m=mkMap(20,20); const things=[{kind:'FoodStore',pos:[12,5],stock:10,res:null}];
  const p=mkPawn(m,2,5); p.sat=0.1;
  let i=0; for(;i<200&&p.done===0;i++) tick(p,things);
  T('eats within 200 ticks', p.done>0, 'ticks='+i);
  T('food rose above 0.1', p.sat>0.1, 'sat='+p.sat.toFixed(3));
  T('store went down', things[0].stock<10, 'stock='+things[0].stock.toFixed(2));
  T('ended adjacent to store', H(p.pos,[12,5])<=SQRT2+0.01, 'pos='+p.pos); }
// 2. tired pawn sleeps in a bed and wakes rested
{ const m=mkMap(20,20); const things=[{kind:'Bed',pos:[7,5],stock:-1,res:null}];
  const p=mkPawn(m,5,5); p.ene=0.05;
  let i=0; for(;i<400&&p.done===0;i++) tick(p,things);
  T('sleep job completed', p.done>0, 'ticks='+i);
  T('wakes rested', p.ene>0.9, 'ene='+p.ene.toFixed(3)+' ticks='+i);
  T('is awake after the job ends', p.asleep===false); }
// 3. worst need wins
{ const m=mkMap(20,20); const things=[{kind:'FoodStore',pos:[6,5],stock:10,res:null},{kind:'WaterSource',pos:[4,5],stock:10,res:null}];
  const p=mkPawn(m,5,5); p.sat=0.39; p.hyd=0.05;
  T('thirst beats hunger', giveJob(p,things).def==='Drink'); }
// 4. reserved store not double-handed
{ const m=mkMap(20,20); const only={kind:'FoodStore',pos:[6,5],stock:10,res:null}; const things=[only];
  const a=mkPawn(m,5,5),b=mkPawn(m,5,6); a.sat=0.1;b.sat=0.1;
  const ja=giveJob(a,things); T('first gets Eat', ja.def==='Eat'); T('store reserved', only.res===a);
  T('second not sent to it', giveJob(b,things).def!=='Eat'); }
// 5. empty store ignored
{ const m=mkMap(20,20); const things=[{kind:'FoodStore',pos:[6,5],stock:0,res:null}];
  const p=mkPawn(m,5,5); p.sat=0.1; T('empty store ignored', giveJob(p,things).def!=='Eat'); }
// 6. no stored water -> drink from lake, standing on the bank
{ const m=mkMap(20,20); m.t[5*20+10]='water'; const things=[];
  const p=mkPawn(m,5,5); p.hyd=0.1; const j=giveJob(p,things);
  T('drinks from lake', j.def==='Drink'&&j.thing===null);
  T('stands on the bank', m.walkable(j.cell[0],j.cell[1])); }
// 7. unreachable store fails and the pawn stays put
{ const m=mkMap(20,20);
  for(let x=9;x<=13;x++){m.walls.add(x+',3');m.walls.add(x+',7');}
  for(let y=3;y<=7;y++){m.walls.add('9,'+y);m.walls.add('13,'+y);}
  const things=[{kind:'FoodStore',pos:[11,5],stock:10,res:null}];
  const p=mkPawn(m,2,5); p.sat=0.1;
  for(let i=0;i<50;i++) tick(p,things);
  T('failure recorded', p.failed!==null);
  T('pawn stayed put', p.pos[0]===2&&p.pos[1]===5, 'pos='+p.pos); }
// 8. dead pawn releases its bed
{ const m=mkMap(20,20); const bed={kind:'Bed',pos:[6,5],stock:-1,res:null}; const things=[bed];
  const p=mkPawn(m,5,5); p.ene=0.05; tick(p,things);
  T('bed claimed', bed.res===p);
  p.sat=0;p.hyd=0; let i=0; for(;i<1000&&p.alive;i++) tick(p,things);
  T('pawn died', !p.alive, 'ticks='+i); T('bed released', bed.res===null); }
// 9. rough ground is slower
{ const cross=(kind)=>{ const m=mkMap(12,3); m.t.fill(kind); const p=mkPawn(m,0,1);
    if(!startPath(p,[8,1]))return -1; let t=0; while(p.path&&t<500){pathTick(p);t++;} return (p.pos[0]===8&&p.pos[1]===1)?t:-1; };
  const soil=cross('soil'), forest=cross('forest');
  T('reaches dest on soil', soil>0, 'soil='+soil); T('reaches dest in forest', forest>0, 'forest='+forest);
  T('forest is slower', forest>soil, soil+' vs '+forest); }
// 10. wall raised across the path forces a reroute
{ const m=mkMap(14,7); const p=mkPawn(m,1,3);
  T('initial path found', startPath(p,[12,3]));
  for(let i=0;i<6;i++)pathTick(p);
  for(let y=0;y<7;y++) if(y!==6) m.walls.add('8,'+y);
  let t=0; while(p.path&&t<600){pathTick(p);t++;}
  T('reroutes to the gap and arrives', p.pos[0]===12&&p.pos[1]===3, 'pos='+p.pos); }

console.log(fail? `\n${fail} FAILED, ${pass} passed` : `all ${pass} job-loop checks pass`);
process.exit(fail?1:0);
