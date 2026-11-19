import asyncio, json
from playwright.async_api import async_playwright

RUN = r"""
(arg) => {
  const [pre] = arg;
  __agon.setState({ playing: false });
  for (let i = 0; i < pre; i++) __agon.advance();
  const dump = () => {
    const out = {};
    const walk = (v, p) => {
      if (v === null || v === undefined) { out[p] = String(v); return; }
      if (typeof v === 'number') { out[p] = Number.isInteger(v) ? v : Math.round(v*1e12)/1e12; return; }
      if (typeof v !== 'object') { out[p] = v; return; }
      if (v instanceof Set) { out[p+'#s'] = [...v].join(','); return; }
      if (v instanceof Map) { out[p+'#m'] = [...v.entries()].map(e=>e[0]+'='+e[1]).join(','); return; }
      if (Array.isArray(v)) { out[p+'.len']=v.length; for (let i=0;i<v.length;i++) walk(v[i], p+'['+i+']'); return; }
      for (const kk of Object.keys(v).sort()) walk(v[kk], p+'.'+kk);
    };
    walk(__agon.state, 's');
    return out;
  };
  const a = dump();
  __agon.advance();
  const b = dump();
  return JSON.stringify({ before: a, after: b });
}
"""

async def run(pre):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1000,'height':640})
        await pg.add_init_script("window.AGON_P={paused:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(120):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(500)
        r = await pg.evaluate(RUN, [pre])
        await b.close()
        return json.loads(r)

def diff(a,b):
    ks=set(a)|set(b)
    return sorted(k for k in ks if a.get(k,'<m>')!=b.get(k,'<m>'))

async def main():
    runs=[await run(1092) for _ in range(4)]
    # before-state must be identical everywhere
    for j in range(1,4):
        d = diff(runs[0]['before'], runs[j]['before'])
        print(f'tick1092 run0 vs run{j}: {len(d)} differing field(s)')
    print()
    for j in range(1,4):
        d = diff(runs[0]['after'], runs[j]['after'])
        print(f'tick1093 run0 vs run{j}: {len(d)} differing field(s)')
        for k in d[:40]:
            print(f'    {k}: {runs[0]["after"].get(k,"<m>")!r} vs {runs[j]["after"].get(k,"<m>")!r}')
        if d: break

asyncio.run(main())
