# Determinism acceptance test. Fresh pages, no instrumentation beyond one exact hash at the end.
# Exact: String(number) is the shortest round-trip form, so nothing is rounded away.
import asyncio, sys
from playwright.async_api import async_playwright

RUN = r"""
(arg) => {
  const [k, seed, biome] = arg;
  if (seed) { __agon.setWorld(undefined, undefined, biome, seed); }
  __agon.setState({ playing: false });
  for (let i = 0; i < k; i++) __agon.advance();
  const parts = [];
  const ser = (v, d) => {
    if (d > 8) { parts.push('~'); return; }
    if (v === null || v === undefined) { parts.push(String(v)); return; }
    if (typeof v !== 'object') { parts.push(String(v)); return; }
    if (v instanceof Set) { parts.push('S'); for (const x of v) ser(x, d+1); return; }
    if (v instanceof Map) { parts.push('M'); for (const e of v) { ser(e[0], d+1); ser(e[1], d+1); } return; }
    if (Array.isArray(v)) { parts.push('['+v.length); for (let i=0;i<v.length;i++) ser(v[i], d+1); return; }
    const ks = Object.keys(v).sort(); parts.push('{'+ks.length);
    for (const kk of ks) { parts.push(kk); ser(v[kk], d+1); }
  };
  ser(__agon.state, 0);
  const h=(x)=>{let n=5381;for(let i=0;i<x.length;i++){n=((n*33)^x.charCodeAt(i))>>>0;}return n.toString(36);};
  const s = __agon.state;
  return { h: h(parts.join('\u0001')), tick: s.tick,
           alive: (s.agents||[]).filter(a=>a.alive&&!a.fac).length,
           built: ((s.store||{}).built||[]).length, techs: ((s.store||{}).techs||[]).length };
}
"""

async def run(ticks, seed, biome):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1000,'height':640})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.add_init_script("window.AGON_P={paused:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(400):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(600)
        if seed: await pg.wait_for_timeout(1200)
        r = await pg.evaluate(RUN, [ticks, seed, biome])
        await b.close()
        return r, errs

async def main():
    cases = [(6000, None, None), (9000, None, None), (4000, 21, 'temperate'), (4000, 7, 'tundra')]
    bad = 0
    for ticks, seed, biome in cases:
        rows=[]
        for i in range(3):
            r,e = await run(ticks, seed, biome); rows.append(r)
            if e: print('   page error:', e[:1], flush=True)
        d = {r['h'] for r in rows}
        lbl = f"{ticks} ticks" + (f", seed {seed} {biome}" if seed else ", default world")
        ok = len(d)==1
        if not ok: bad += 1
        print(f"{'PASS' if ok else 'FAIL'}  {lbl}: {len(d)} distinct exact hash(es) over 3 pages "
              f"(alive={[r['alive'] for r in rows]} built={[r['built'] for r in rows]} tick={rows[0]['tick']})", flush=True)
    print('\nRESULT:', 'all deterministic' if bad==0 else f'{bad} case(s) still diverge')
    sys.exit(1 if bad else 0)

asyncio.run(main())
