import asyncio
from playwright.async_api import async_playwright

RUN = r"""
(k) => {
  __agon.setState({ playing: false });
  for (let i = 0; i < k; i++) __agon.advance();
  const h = (str) => { let x=5381; for (let i=0;i<str.length;i++){ x=((x*33)^str.charCodeAt(i))>>>0; } return x.toString(36); };
  const rnd = (key,v) => (typeof v === 'number' && !Number.isInteger(v)) ? Math.round(v*1e6)/1e6 : v;
  const s = __agon.state, E = __agon.E || {};
  const alive = (s.agents||[]).filter(a=>a.alive && !a.fac).length;
  return { tick: s.tick, alive,
           st: h(JSON.stringify(s, rnd)),
           walls: E.WALLS?E.WALLS.size:-1, built: ((s.store||{}).built||[]).length,
           techs: ((s.store||{}).techs||[]).length };
}
"""

async def run(ticks):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1000,'height':640})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.add_init_script("window.AGON_P={paused:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(100):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(500)
        r = await pg.evaluate(RUN, ticks)
        await b.close()
        return r, errs

async def main():
    for ticks in (1500, 3000):
        rows=[]
        for i in range(5):
            r,e = await run(ticks); rows.append(r)
            print(f"  ticks={ticks} run{i}: alive={r['alive']} walls={r['walls']} built={r['built']} techs={r['techs']} st={r['st']}")
            if e: print('    err:', e[:1])
        d = {r['st'] for r in rows}
        print(f"=> {ticks} ticks, 5 uninstrumented pages: {len(d)} distinct state hash(es)", "-- DETERMINISTIC" if len(d)==1 else "-- DIVERGES")
        print()

asyncio.run(main())
