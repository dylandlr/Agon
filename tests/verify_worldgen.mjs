// Independent re-implementation of WorldGenerator.cs, to check its behaviour.
const clamp01=v=>v<0?0:v>1?1:v, lerp=(a,b,t)=>a+(b-a)*t;
function mkNoise(seed){ const S=(seed>>>0)===0?0x9E3779B9:(seed>>>0);
  const hash=(x,y,c=0)=>{ let h=S>>>0;
    h=(h^Math.imul(x>>>0,0x8DA6B343))>>>0; h=(h^Math.imul(y>>>0,0xD8163841))>>>0; h=(h^Math.imul(c>>>0,0xCB1AB31F))>>>0;
    h=(h^(h>>>16))>>>0; h=Math.imul(h,0x7FEB352D)>>>0; h=(h^(h>>>15))>>>0; h=Math.imul(h,0x846CA68B)>>>0; h=(h^(h>>>16))>>>0;
    return (h & 0xFFFFFF)/0x1000000; };
  const value=(x,y,c=0)=>{ const x0=Math.floor(x),y0=Math.floor(y),fx=x-x0,fy=y-y0;
    const sx=fx*fx*(3-2*fx), sy=fy*fy*(3-2*fy);
    const a=lerp(hash(x0,y0,c),hash(x0+1,y0,c),sx), b=lerp(hash(x0,y0+1,c),hash(x0+1,y0+1,c),sx);
    return lerp(a,b,sy); };
  const fractal=(x,y,oct=4,pers=0.5,c=0)=>{ let s=0,a=1,f=1,t=0;
    for(let i=0;i<oct;i++){ s+=value(x*f,y*f,c+i*31)*a; t+=a; a*=pers; f*=2; } return t>0?s/t:0; };
  const ridged=(x,y,oct=4,c=0)=>1-Math.abs(fractal(x,y,oct,0.5,c)*2-1);
  return {hash,value,fractal,ridged}; }
const BIOMES={
 Temperate:{ground:0.38,tree:0.62,temp:11,swing:14,water:0.10,rock:0.05},
 Desert:{ground:0.86,tree:0.97,temp:28,swing:16,water:0.01,rock:0.12},
 Tundra:{ground:0.74,tree:0.93,temp:-8,swing:18,water:0.05,rock:0.10},
 Savanna:{ground:0.60,tree:0.88,temp:24,swing:9,water:0.03,rock:0.07},
 Jungle:{ground:0.20,tree:0.40,temp:26,swing:5,water:0.14,rock:0.03},
 Taiga:{ground:0.34,tree:0.52,temp:1,swing:20,water:0.11,rock:0.08}};
const ROCK=0.70, CLIFF=0.82, CR=6;
const T={Soil:'soil',Grass:'grass',Forest:'forest',Rock:'rock',Water:'water',Sand:'sand'};
function gen(W,H,seed,bname){
  const B=BIOMES[bname], n=mkNoise(seed), idx=(x,y)=>y*W+x;
  const el=new Float64Array(W*H), f=5/Math.max(W,H);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    let e=n.fractal(x*f,y*f,5,0.5); const r=n.ridged(x*f*0.5,y*f*0.5,3,977);
    el[idx(x,y)]=clamp01(lerp(e,r,0.28)); }
  // waterline by quantile
  let wl=-1; if(B.water>0){ const s=Array.from(el).sort((a,b)=>a-b);
    wl=s[Math.min(s.length-1,Math.max(0,Math.floor(B.water*s.length)))]; }
  const ter=new Array(W*H).fill(T.Soil), rockEd=new Set();
  const cs=9/Math.max(W,H);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){ const e=el[idx(x,y)];
    if(e<=wl){ter[idx(x,y)]=T.Water;continue;}
    if(e>=CLIFF){ter[idx(x,y)]=T.Rock;rockEd.add(x+','+y);continue;}
    if(e>=ROCK){ter[idx(x,y)]=T.Rock;continue;}
    const cov=n.fractal(x*cs,y*cs,4,0.5,313);
    const shore=e<wl+0.03 && B.ground>0.5;
    ter[idx(x,y)]= shore?T.Sand : cov>B.tree?T.Forest : cov>B.ground?T.Grass : T.Soil; }
  // camp site
  let best=[Math.floor(W/2),Math.floor(H/2)], bs=Infinity, m=CR+2;
  for(let y=m;y<H-m;y++)for(let x=m;x<W-m;x++){ const c=el[idx(x,y)];
    if(c<=wl||c>=ROCK)continue; let rough=0,blocked=false;
    for(let dy=-CR;dy<=CR&&!blocked;dy++)for(let dx=-CR;dx<=CR;dx++){
      const e=el[idx(x+dx,y+dy)]; if(e>=CLIFF){blocked=true;break;} rough+=Math.abs(e-c); }
    if(blocked)continue;
    const dry=Math.max(0,0.12-(c-wl))*40, sc=rough+dry;
    if(sc<bs){bs=sc;best=[x,y];} }
  // clear camp
  for(let dy=-CR;dy<=CR;dy++)for(let dx=-CR;dx<=CR;dx++){
    const x=best[0]+dx,y=best[1]+dy; if(x<0||y<0||x>=W||y>=H)continue;
    rockEd.delete(x+','+y);
    const cov=n.fractal(x*cs,y*cs,4,0.5,313);
    ter[idx(x,y)]= cov>B.ground?T.Grass:T.Soil; }
  // boulders
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    if(Math.abs(x-best[0])<=CR&&Math.abs(y-best[1])<=CR)continue;
    if(rockEd.has(x+','+y))continue;
    const t=ter[idx(x,y)]; if(t===T.Water||t===T.Rock)continue;
    if(n.hash(x,y,5501) < B.rock*0.25) rockEd.add(x+','+y); }
  return {W,H,el,wl,ter,rockEd,camp:best,idx,B};
}
const share=(w,k)=>w.ter.filter(t=>t===k).length/(w.W*w.H);
let pass=0,fail=0; const T2=(n,c,x)=>{ if(c)pass++; else {fail++;console.log('FAIL: '+n+(x!==undefined?'  ['+x+']':''));} };

// determinism
{ const a=gen(80,50,3,'Temperate'), b=gen(80,50,3,'Temperate');
  T2('same seed gives the same map', a.ter.join()===b.ter.join() && a.camp.join()===b.camp.join());
  const c=gen(80,50,4,'Temperate');
  T2('a different seed gives a different map', a.ter.join()!==c.ter.join()); }
// water share honours the biome
for (const [bn,want] of [['Temperate',0.10],['Jungle',0.14],['Desert',0.01],['Savanna',0.03]]) {
  const w=gen(80,50,7,bn), got=share(w,T.Water);
  T2(bn+' water share ~'+want, Math.abs(got-want)<0.02, 'got '+got.toFixed(3)); }
// biome separation
{ const d=gen(80,50,11,'Desert'), j=gen(80,50,11,'Jungle');
  T2('desert is barer than jungle', share(d,T.Soil)>share(j,T.Soil), share(d,T.Soil).toFixed(2)+' vs '+share(j,T.Soil).toFixed(2));
  T2('jungle has far more forest', share(j,T.Forest)>share(d,T.Forest)+0.2, share(j,T.Forest).toFixed(2)+' vs '+share(d,T.Forest).toFixed(2)); }
// camp quality, across many seeds and biomes
{ let bad=[], bareCamps=0, n=0;
  for(const bn of Object.keys(BIOMES)) for(let s=1;s<=8;s++){ n++;
    const w=gen(80,50,s,bn); const [cx,cy]=w.camp;
    let water=0,rock=0,forest=0,boul=0,soil=0,grass=0,tot=0;
    for(let dy=-CR;dy<=CR;dy++)for(let dx=-CR;dx<=CR;dx++){
      const x=cx+dx,y=cy+dy,t=w.ter[w.idx(x,y)]; tot++;
      if(t===T.Water)water++; if(t===T.Rock)rock++; if(t===T.Forest)forest++;
      if(t===T.Soil)soil++; if(t===T.Grass)grass++;
      if(w.rockEd.has(x+','+y))boul++; }
    if(water||rock||forest||boul) bad.push(bn+'#'+s+' w'+water+' r'+rock+' f'+forest+' b'+boul);
    if(grass===0 && w.B.ground<0.5) bareCamps++;  // a green biome should not give a bare camp
  }
  T2('camp is always buildable (no water, rock, forest or boulders)', bad.length===0, bad.slice(0,3).join(' | '));
  T2('green biomes do not get a bare dirt camp', bareCamps===0, bareCamps+' bare of '+n); }
// camp is on reasonably flat ground
{ let worst=0; for(let s=1;s<=8;s++){ const w=gen(80,50,s,'Temperate'); const [cx,cy]=w.camp;
    const c=w.el[w.idx(cx,cy)]; let mx=0;
    for(let dy=-CR;dy<=CR;dy++)for(let dx=-CR;dx<=CR;dx++) mx=Math.max(mx,Math.abs(w.el[w.idx(cx+dx,cy+dy)]-c));
    worst=Math.max(worst,mx); }
  T2('camp ground is flat (max local relief < 0.25)', worst<0.25, 'worst '+worst.toFixed(3)); }
// camp is inland, away from the map edge
{ let ok=true; for(const bn of Object.keys(BIOMES)) for(let s=1;s<=5;s++){ const w=gen(80,50,s,bn);
    if(w.camp[0]<CR+2||w.camp[1]<CR+2||w.camp[0]>=80-CR-2||w.camp[1]>=50-CR-2) ok=false; }
  T2('camp keeps clear of the map edge', ok); }
// elevation spread is sane (not all flat, not all extremes)
{ const w=gen(80,50,3,'Temperate'); const a=Array.from(w.el);
  const mn=Math.min(...a), mx=Math.max(...a), avg=a.reduce((p,c)=>p+c,0)/a.length;
  T2('elevation has range', mx-mn>0.35, (mx-mn).toFixed(2));
  T2('elevation is not degenerate', avg>0.2&&avg<0.8, avg.toFixed(2)); }
// map size changes landform size, not count
{ const small=gen(60,40,5,'Temperate'), big=gen(120,80,5,'Temperate');
  const peaks=(w)=>{let n=0; for(let y=1;y<w.H-1;y++)for(let x=1;x<w.W-1;x++){const e=w.el[w.idx(x,y)];
    let top=true; for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]) if(w.el[w.idx(x+dx,y+dy)]>=e)top=false;
    if(top)n++;} return n;};
  const ps=peaks(small), pb=peaks(big);
  T2('bigger map does not simply get proportionally more peaks', pb < ps*4, ps+' vs '+pb); }

console.log(fail? `\n${fail} FAILED, ${pass} passed` : `all ${pass} worldgen checks pass`);
process.exit(fail?1:0);
