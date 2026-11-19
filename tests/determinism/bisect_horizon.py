import asyncio, json
from playwright.async_api import async_playwright

RUN = r"""
(k) => {
  __agon.setState({ playing: false });
  for (let i = 0; i < k; i++) __agon.advance();
  const out = {};
  const walk = (v, p) => {
    if (v === null || v === undefined) { out[p] = String(v); return; }
    if (typeof v === 'number') { out[p] = Number.isInteger(v) ? v : Math.round(v*1e6)/1e6; return; }
    if (typeof v !== 'object') { out[p] = v; return; }
    if (v instanceof Set) { out[p+'#set'] = [...v].join(','); return; }
    if (v instanceof Map) { out[p+'#map'] = [...v.entries()].map(e=>e[0]+'='+e[1]).join(','); return; }
    if (Array.isArray(v)) { out[p+'.len'] = v.length; for (let i=0;i<v.length;i++) walk(v[i], p+'['+i+']'); return; }
    for (const kk of Object.keys(v).sort()) walk(v[kk], p+'.'+kk);
  };
  walk(__agon.state, 's');
  return JSON.stringify(out);
}
"""

async def run(ticks):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1000,'height':640})
        await pg.add_init_script("window.AGON_P={paused:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(100):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(500)
        r = await pg.evaluate(RUN, ticks)
        await b.close()
        return json.loads(r)

def diff(a,b):
    ks = set(a)|set(b)
    return [k for k in ks if a.get(k,'<m>')!=b.get(k,'<m>')]

async def main():
    for ticks in (950, 1000, 1040, 1070):
        ds=[await run(ticks) for _ in range(6)]
        pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
        res=[(i,j,diff(ds[i],ds[j])) for i,j in pairs]
        print(f'ticks={ticks:4d}  diffs: ' + '  '.join(f'{i}v{j}={len(d)}' for i,j,d in res))
        nz=[r for r in res if r[2]]
        if nz:
            i,j,d = min(nz, key=lambda r: len(r[2]))
            print(f'  smallest nonzero: {i}v{j} with {len(d)} field(s)')
            for k in sorted(d)[:30]:
                print(f'    {k}: {ds[i].get(k,"<m>")!r} vs {ds[j].get(k,"<m>")!r}')
            return

asyncio.run(main())
