// Independent re-implementation of PathFinder.cs to check its rules and the test expectations.
const SQRT2=1.41421356;
function mk(w,h,rows){ // rows: array of strings, '#'=wall, '.'=open, '~'=water(unwalkable), 'f'=forest(cost1.6)
  return {w,h,rows:rows.slice().reverse()}; }  // row 0 = bottom
const at=(m,x,y)=> (x<0||y<0||x>=m.w||y>=m.h) ? '#' : m.rows[y][x];
const walk=(m,x,y)=>{const c=at(m,x,y); return c!=='#'&&c!=='~';};
const cost=(m,x,y)=>{const c=at(m,x,y); return c==='f'?1.6:1;};
const H=(ax,ay,bx,by)=>{const dx=Math.abs(ax-bx),dy=Math.abs(ay-by),mn=Math.min(dx,dy);return (dx+dy-2*mn)+mn*SQRT2;};
const N=[[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]];
function astar(m,sx,sy,dx,dy){
  if(!walk(m,dx,dy)) return null;
  if(sx===dx&&sy===dy) return [];
  const key=(x,y)=>y*m.w+x, g=new Map(), par=new Map(), closed=new Set();
  const open=[[H(sx,sy,dx,dy),key(sx,sy)]]; g.set(key(sx,sy),0);
  let exp=0;
  while(open.length){
    open.sort((a,b)=>a[0]-b[0]);
    const [,cur]=open.shift();
    if(closed.has(cur))continue; closed.add(cur);
    const cx=cur%m.w, cy=(cur-cx)/m.w;
    if(cx===dx&&cy===dy){ const path=[]; let a=cur; while(par.has(a)){path.push([a%m.w,(a-a%m.w)/m.w]); a=par.get(a);} return path.reverse(); }
    if(++exp>20000) return null;
    for(const [ox,oy] of N){
      const nx=cx+ox, ny=cy+oy;
      if(!walk(m,nx,ny))continue;
      const diag=ox!==0&&oy!==0;
      if(diag && (!walk(m,cx+ox,cy)||!walk(m,cx,cy+oy))) continue;  // no corner cutting
      const nk=key(nx,ny); if(closed.has(nk))continue;
      const t=g.get(cur)+(diag?SQRT2:1)*cost(m,nx,ny);
      if(t < (g.has(nk)?g.get(nk):Infinity)){ g.set(nk,t); par.set(nk,cur); open.push([t+H(nx,ny,dx,dy),nk]); }
    }
  }
  return null;
}
let pass=0,fail=0; const T=(n,c)=>{ if(c)pass++; else {fail++;console.log('FAIL: '+n);} };

// 1. straight line on open ground
{ const m=mk(8,3,['........','........','........']);
  const p=astar(m,0,1,7,1); T('straight path found', p&&p.length===7);
  T('straight path ends at dest', p&&p[p.length-1][0]===7&&p[p.length-1][1]===1);
  T('start cell excluded', p&&!(p[0][0]===0&&p[0][1]===1)); }
// 2. wall in the way -> goes around
{ const m=mk(7,5,['.......','...#...','...#...','...#...','.......']);
  const p=astar(m,0,2,6,2);
  T('path routes around a wall', p&&p.length>=6);
  T('path never enters the wall', p&&p.every(([x,y])=>at(m,x,y)!=='#'));
  T('path detours off the straight line', p&&p.some(([x,y])=>y!==2)); }
// 3. fully sealed room -> no path
{ const m=mk(7,7,['.......','.#####.','.#...#.','.#.#.#.','.#...#.','.#####.','.......']);
  // target inside a sealed box (3,3 is a wall; use 3,4 interior)
  const p=astar(m,0,0,3,4); T('sealed room unreachable', p===null); }
// 4. NO CORNER CUTTING: two walls meeting diagonally must not be slipped through
{ const m=mk(3,3,['...','.#.','..#']);
  // from (1,0) to (2,1): diagonal step would pass between walls at (1,1) and (2,0)? set up precisely:
  const m2=mk(3,3,['.#.','#..','...']);   // row0(bottom)='...', row1='#..', row2='.#.'
  // walls at (0,1) and (1,2). From (0,2) to (1,1) is diagonal through (0,1)# and (1,2)#
  const p=astar(m2,0,2,1,1);
  T('diagonal between two walls is refused', p===null || !(p.length===1));
}
// 5. destination unwalkable
{ const m=mk(5,3,['.....','..#..','.....']);
  T('wall destination refused', astar(m,0,0,2,1)===null); }
// 6. water is not walkable
{ const m=mk(5,3,['.....','..~..','.....']);
  T('water destination refused', astar(m,0,0,2,1)===null); }
// 7. terrain cost: path prefers open ground over forest
{ const m=mk(7,3,['.......','.fffff.','.......']);
  const p=astar(m,0,1,6,1);
  const fTiles=p.filter(([x,y])=>at(m,x,y)==='f').length;
  T('path avoids costly forest when it can', fTiles<=1); }
// 8. octile heuristic is admissible (never exceeds true cost on open ground)
{ const m=mk(10,10,Array(10).fill('..........'));
  let ok=true;
  for(let i=0;i<40;i++){ const ax=i%10, ay=(i*3)%10, bx=(i*7)%10, by=(i*5)%10;
    const p=astar(m,ax,ay,bx,by); if(!p)continue;
    let g=0,cx=ax,cy=ay; for(const [x,y] of p){ g+=(x!==cx&&y!==cy)?SQRT2:1; cx=x;cy=y; }
    if(H(ax,ay,bx,by) > g+1e-6) ok=false; }
  T('heuristic never overestimates', ok); }
// 9. doorway: a one-cell gap is used
{ // a closed box with a single doorway in its south wall, at x=3
  const m3=mk(7,5,['.......','.#####.','.#...#.','.##.##.','.......']);
  const p=astar(m3,0,0,3,2);
  T('path uses a one-cell doorway', p!==null);
  T('path passes through the doorway cell', p&&p.some(([x,y])=>x===3&&y===1));
  // seal that doorway and the interior must become unreachable
  const m4=mk(7,5,['.......','.#####.','.#...#.','.#####.','.......']);
  T('sealing the doorway makes it unreachable', astar(m4,0,0,3,2)===null); }

console.log(fail? `\n${fail} FAILED, ${pass} passed` : `all ${pass} A* checks pass`);
process.exit(fail?1:0);
