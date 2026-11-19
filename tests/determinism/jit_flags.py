import asyncio, sys
from playwright.async_api import async_playwright

RUN = r"""
(k) => {
  __agon.setState({ playing: false });
  for (let i = 0; i < k; i++) __agon.advance();
  const out = {};
  const walk = (v, p) => {
    if (v === null || v === undefined) { out[p] = String(v); return; }
    if (typeof v === 'number') { out[p] = Number.isInteger(v) ? v : Math.round(v*1e9)/1e9; return; }
    if (typeof v !== 'object') { out[p] = v; return; }
    if (v instanceof Set) { out[p+'#s'] = [...v].join(','); return; }
    if (v instanceof Map) { out[p+'#m'] = [...v.entries()].map(e=>e[0]+'='+e[1]).join(','); return; }
    if (Array.isArray(v)) { out[p+'.len']=v.length; for (let i=0;i<v.length;i++) walk(v[i], p+'['+i+']'); return; }
    for (const kk of Object.keys(v).sort()) walk(v[kk], p+'.'+kk);
  };
  walk(__agon.state, 's');
  const h = (str)=>{let x=5381;for(let i=0;i<str.length;i++){x=((x*33)^str.charCodeAt(i))>>>0;}return x.toString(36);};
  return h(JSON.stringify(out));
}
"""

async def run(ticks, flags):
    async with async_playwright() as p:
        args = ['--js-flags='+flags] if flags else []
        b = await p.chromium.launch(args=args)
        pg = await b.new_page(viewport={'width':1000,'height':640})
        await pg.add_init_script("window.AGON_P={paused:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(140):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(500)
        r = await pg.evaluate(RUN, ticks)
        await b.close()
        return r

async def main():
    for label, flags in [('--no-opt', '--no-opt'), ('default', None)]:
        hs=[]
        for i in range(4):
            h = await run(1500, flags); hs.append(h)
            print(f"  {label:9s} run{i}: {h}")
        d=set(hs)
        print(f"=> {label}: {len(d)} distinct hash(es) over 4 pages", "-- DETERMINISTIC" if len(d)==1 else "-- DIVERGES")
        print()

asyncio.run(main())
