// Independent re-implementation of the C# linking rules, to check the test expectations.
const W=16,H=16;
const NONE=0,WALL=4,DOOR=8,ROCK=2,EDGE=1, WALLLIKE=EDGE|ROCK|WALL|DOOR;
const CARD=[[0,1],[1,0],[0,-1],[-1,0]];           // Up,Right,Down,Left
const DIAG=[[-1,-1],[-1,1],[1,1],[1,-1]];         // SW,NW,NE,SE
const SW=0,NW=1,NE=2,SE=3;
function mk(){ return {g:new Map()}; }
const k=(x,y)=>x+','+y;
function set(m,x,y,f){ m.g.set(k(x,y),f); }
function flags(m,x,y){ if(x<0||y<0||x>=W||y>=H) return EDGE; return m.g.get(k(x,y))||NONE; }
function link(m,x,y,lf){ return (flags(m,x,y)&lf)!==0; }
function mask(m,x,y,lf){ let n=0,b=1; for(const[dx,dy]of CARD){ if(link(m,x+dx,y+dy,lf)) n+=b; b*=2; } return n; }
function fill(m,x,y,i,lf){ const[dx,dy]=DIAG[i];
  if(!link(m,x+dx,y+dy,lf)) return false;
  return link(m,x+dx,y,lf) && link(m,x,y+dy,lf); }
function nfill(m,x,y,lf){ let n=0; for(let i=0;i<4;i++) if(fill(m,x,y,i,lf)) n++; return n; }
let pass=0,fail=0;
const T=(name,cond)=>{ if(cond){pass++;} else {fail++; console.log('FAIL: '+name);} };

// --- LinkMask bit order
{ const m=mk(); set(m,5,5,WALL);
  T('isolated=None', mask(m,5,5,WALLLIKE)===0);
  set(m,5,6,WALL); T('Up=1', mask(m,5,5,WALLLIKE)===1);
  set(m,6,5,WALL); T('Right=2 -> 3', mask(m,5,5,WALLLIKE)===3);
  set(m,5,4,WALL); T('Down=4 -> 7', mask(m,5,5,WALLLIKE)===7);
  set(m,4,5,WALL); T('Left=8 -> 15', mask(m,5,5,WALLLIKE)===15); }
// --- doors/rock link, ground doesn't
{ const m=mk(); set(m,5,5,WALL); set(m,5,6,DOOR); set(m,6,5,ROCK);
  const l=mask(m,5,5,WALLLIKE);
  T('door links', (l&1)!==0); T('rock links', (l&2)!==0);
  T('ground S no', (l&4)===0); T('ground W no', (l&8)===0); }
// --- map edge
{ const m=mk(); set(m,0,0,WALL); const l=mask(m,0,0,WALLLIKE);
  T('edge S links',(l&4)!==0); T('edge W links',(l&8)!==0); }
// --- atlas uv
{ let ok=true,ovl=false; const rects=[];
  for(let i=0;i<16;i++){ const x=(i%4)*0.25+0.03125, y=Math.floor(i/4)*0.25+0.03125;
    if(x<0||y<0||x+0.1875>1.0000001||y+0.1875>1.0000001) ok=false; rects.push([x,y,0.1875,0.1875]); }
  for(let i=0;i<16;i++)for(let j=i+1;j<16;j++){ const[a,b]=[rects[i],rects[j]];
    if(a[0]<b[0]+b[2]&&b[0]<a[0]+a[2]&&a[1]<b[1]+b[3]&&b[1]<a[1]+a[3]) ovl=true; }
  T('uv in sheet', ok); T('uv no overlap', !ovl); }

// --- CornerFiller cases
{ const m=mk(); [[5,5],[6,5],[5,6],[6,6]].forEach(([x,y])=>set(m,x,y,WALL));
  T('2x2 SW tile fills NE', fill(m,5,5,NE,WALLLIKE));
  T('2x2 SE tile fills NW', fill(m,6,5,NW,WALLLIKE));
  T('2x2 NW tile fills SE', fill(m,5,6,SE,WALLLIKE));
  T('2x2 NE tile fills SW', fill(m,6,6,SW,WALLLIKE));
  T('2x2 (5,5) exactly 1', nfill(m,5,5,WALLLIKE)===1);
  T('2x2 (6,6) exactly 1', nfill(m,6,6,WALLLIKE)===1); }
{ const m=mk(); for(let x=3;x<=9;x++) set(m,x,5,WALL);
  let all0=true; for(let x=4;x<=8;x++) if(nfill(m,x,5,WALLLIKE)!==0) all0=false;
  T('straight run fills 0', all0); }
{ const m=mk(); set(m,5,5,WALL); set(m,4,5,WALL); set(m,5,4,WALL);
  T('L corner fills 0', nfill(m,5,5,WALLLIKE)===0); }
{ const m=mk(); set(m,5,5,WALL); set(m,6,6,WALL);
  T('diagonal touch (5,5) 0', nfill(m,5,5,WALLLIKE)===0);
  T('diagonal touch (6,6) 0', nfill(m,6,6,WALLLIKE)===0); }
{ const m=mk(); for(let x=3;x<=8;x++){ set(m,x,5,WALL); set(m,x,6,WALL); }
  T('thick NE', fill(m,5,5,NE,WALLLIKE));
  T('thick NW', fill(m,5,5,NW,WALLLIKE));
  T('thick SE false', !fill(m,5,5,SE,WALLLIKE)); }
{ const m=mk(); set(m,4,5,WALL); set(m,4,6,WALL); set(m,5,5,DOOR); set(m,5,6,DOOR); set(m,6,5,WALL); set(m,6,6,WALL);
  T('door keeps run', fill(m,4,5,NE,WALLLIKE)); }
{ const m=mk(); set(m,0,0,WALL);
  T('map corner fills SW', fill(m,0,0,SW,WALLLIKE)); }

// --- corner quad geometry: four tiles' patches must exactly tile the area around the shared corner
{ const S=0.5, quads=[];
  const org=(cx,cy,dx,dy)=>[cx+(dx>0?S:0), cy+(dy>0?S:0)];
  quads.push(org(5,5,1,1)); quads.push(org(6,5,-1,1)); quads.push(org(5,6,1,-1)); quads.push(org(6,6,-1,-1));
  // union should be exactly the square (5.5,5.5)-(6.5,6.5) centred on the shared corner (6,6)
  const xs=quads.map(q=>q[0]), ys=quads.map(q=>q[1]);
  const okX = Math.min(...xs)===5.5 && Math.max(...xs)===6.0;
  const okY = Math.min(...ys)===5.5 && Math.max(...ys)===6.0;
  const distinct = new Set(quads.map(q=>q.join(','))).size===4;
  T('corner quads tile the shared corner', okX&&okY&&distinct); }

console.log(fail? `\n${fail} FAILED, ${pass} passed` : `\nall ${pass} logic checks pass`);
process.exit(fail?1:0);
