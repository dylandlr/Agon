import asyncio, json
from playwright.async_api import async_playwright

RUN = r"""
(pre) => {
  __agon.setState({ playing: false });
  for (let i = 0; i < pre; i++) __agon.advance();
  const h = (str)=>{let x=5381;for(let i=0;i<str.length;i++){x=((x*33)^str.charCodeAt(i))>>>0;}return x.toString(36);};
  const ser = (v, d) => {
    if (d > 6) return '~';
    if (v === null || v === undefined) return String(v);
    const t = typeof v;
    if (t === 'function') return 'fn';
    if (t === 'number') return Number.isInteger(v) ? String(v) : String(Math.round(v*1e12)/1e12);
    if (t !== 'object') return String(v);
    if (typeof Node !== 'undefined' && v instanceof Node) return 'dom';
    if (v instanceof Set) { try { return 'S['+[...v].map(x=>ser(x,d+1)).join(',')+']'; } catch(e){ return 'S?'; } }
    if (v instanceof Map) { try { return 'M['+[...v.entries()].map(e=>ser(e[0],d+1)+'='+ser(e[1],d+1)).join(',')+']'; } catch(e){ return 'M?'; } }
    if (Array.isArray(v)) { if (v.length > 6000) return 'A'+v.length; return '['+v.map(x=>ser(x,d+1)).join(',')+']'; }
    if (ArrayBuffer.isView(v)) return 'TA'+v.length+':'+h(Array.prototype.join.call(v,','));
    let ks; try { ks = Object.keys(v).sort(); } catch(e) { return 'obj?'; }
    if (ks.length > 4000) return 'O'+ks.length;
    return '{'+ks.map(kk=>kk+':'+ser(v[kk],d+1)).join(',')+'}';
  };
  const res = {};
  for (const kk of Object.keys(__agon).sort()) { try { res['A.'+kk] = h(ser(__agon[kk],0)); } catch(e) { res['A.'+kk]='ERR'; } }
  const E = __agon.E || {};
  for (const kk of Object.keys(E).sort()) { try { res['E.'+kk] = h(ser(E[kk],0)); } catch(e) { res['E.'+kk]='ERR'; } }
  return JSON.stringify(res);
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
        r = await pg.evaluate(RUN, pre)
        await b.close()
        return json.loads(r)

async def main():
    ds=[]
    for i in range(6):
        d = await run(1092); ds.append(d); print('page',i,'keys',len(d))
    allk = sorted(set().union(*[set(d) for d in ds]))
    varying=[]
    for k in allk:
        vals = {d.get(k,'<m>') for d in ds}
        if len(vals)>1: varying.append((k,[d.get(k,'<m>') for d in ds]))
    print(f'\n{len(varying)} of {len(allk)} keys VARY across the 6 pages at tick 1092:')
    for k,v in varying: print('   ', k, v)

asyncio.run(main())
